"""Which printer wins when several are eligible — and what that costs.

The setup is a small mixed farm: three A1s, one of them carrying a four-slot
AMS that is the *only* source of white, the other two loaded with black only.

    printer 1  printer-a      black only
    printer 2  printer-b      AMS: PETG black, PLA white, PLA black, PLA beige
    printer 3  printer-c      black only

The operator's expectation was that a plain black job should go to one of the
single-spool machines, leaving the versatile one free for the job that
genuinely needs it. That is best-fit allocation, and the scheduler does not
implement it: eligible printers are taken in whatever order the database
returns (``_printers_for_model`` issues no ``ORDER BY``), and colour
preferences *rank* by how many overrides match — which favours the printer
with more spools loaded, not fewer.

These tests pin best-fit allocation: among equally eligible printers the one
carrying the FEWEST filaments wins, so the versatile machine stays free for the
job that needs it. The last test is the one that matters — before the change it
failed, with a white job stranded because a black job that had two other
options had taken the only machine that could print white.
"""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import backend.app.models  # noqa: F401 - populate Base.metadata
from backend.app.core.database import Base
from backend.app.models.printer import Printer
from backend.app.services.print_scheduler import PrintScheduler

WHITE = "#FFFFFF"
BLACK = "#161616"

# raw_data as the MQTT client stores it, per printer.
TRAYS = {
    1: [{"tray_type": "PLA", "tray_color": "161616FF"}],
    2: [
        {"tray_type": "PETG", "tray_color": "161616FF"},
        {"tray_type": "PLA", "tray_color": "FFFFFFFF"},
        {"tray_type": "PLA", "tray_color": "161616FF"},
        {"tray_type": "PLA", "tray_color": "D3C5A3FF"},
    ],
    3: [{"tray_type": "PLA", "tray_color": "161616FF"}],
}

NAMES = {1: "printer-a (black only)", 2: "printer-b (AMS, only white)", 3: "printer-c (black only)"}


@pytest.fixture
async def farm():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with session_maker() as db:
        for pid, name in NAMES.items():
            db.add(
                Printer(
                    id=pid,
                    name=name,
                    serial_number=f"A1{pid:013d}",
                    ip_address=f"192.0.2.{pid}",  # RFC 5737 documentation range
                    access_code="x",
                    model="A1",
                    is_active=True,
                )
            )
        await db.commit()

    try:
        yield session_maker
    finally:
        await engine.dispose()


def _status(printer_id):
    return SimpleNamespace(raw_data={"ams": [{"tray": TRAYS[printer_id]}]})


class _Ask:
    """Runs the real matcher against the fake farm, tracking claimed printers.

    ``exclude_ids`` accumulates exactly as ``check_queue`` does within one
    pass: a printer handed to one item is not offered to the next.
    """

    def __init__(self, session_maker):
        self.session_maker = session_maker
        self.scheduler = PrintScheduler()
        self.claimed: set[int] = set()

    async def __call__(self, colour):
        patches = [
            patch("backend.app.services.print_scheduler.printer_manager.is_connected", MagicMock(return_value=True)),
            patch(
                "backend.app.services.print_scheduler.printer_manager.get_status",
                MagicMock(side_effect=_status),
            ),
            patch.object(self.scheduler, "_is_printer_idle", MagicMock(return_value=True)),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            async with self.session_maker() as db:
                printer_id, reason = await self.scheduler._find_idle_printer_for_model(
                    db,
                    "A1",
                    exclude_ids=set(self.claimed),
                    required_filament_types=["PLA"],
                    filament_overrides=[{"type": "PLA", "color": colour}],
                )
        if printer_id is not None:
            self.claimed.add(printer_id)
        return printer_id, reason


@pytest.mark.asyncio
async def test_white_can_only_go_to_the_ams_printer(farm):
    """The hard part of matching works: colour narrows to the one printer that has it."""
    ask = _Ask(farm)
    assert await ask(WHITE) == (2, None)


@pytest.mark.asyncio
async def test_white_first_leaves_the_black_jobs_the_single_spool_machines(farm):
    """The order that happens to work — and only because white was queued first."""
    ask = _Ask(farm)
    assert (await ask(WHITE))[0] == 2
    assert (await ask(BLACK))[0] == 1
    assert (await ask(BLACK))[0] == 3


@pytest.mark.asyncio
async def test_a_black_job_leaves_the_ams_printer_alone(farm):
    """Best fit: a black job takes a single-spool machine, not the AMS.

    All three printers carry black and all three are idle, so every one of
    them is eligible. The AMS printer carries four filaments and the other two
    carry one each — taking it would tie up four times the capability for the
    same result.
    """
    ask = _Ask(farm)
    assert (await ask(BLACK))[0] in {1, 3}
    assert (await ask(BLACK))[0] in {1, 3}
    assert 2 not in ask.claimed, "the AMS printer was consumed by a single-colour job"


@pytest.mark.asyncio
async def test_choice_is_reproducible(farm):
    """Two identical farms answer identically — no dependence on row order."""
    laeufe = []
    for _ in range(2):
        ask = _Ask(farm)
        laeufe.append([(await ask(BLACK))[0], (await ask(BLACK))[0], (await ask(WHITE))[0]])
    assert laeufe[0] == laeufe[1]


@pytest.mark.asyncio
async def test_two_black_jobs_no_longer_strand_the_white_one(farm):
    """The regression this change exists to prevent.

    Three jobs in the order a shop actually receives them — colour is decided
    by whoever ordered, not by what would suit the farm. Before best-fit the
    black jobs took the AMS printer and the white job was left with no printer
    at all, while a black-only machine sat idle.
    """
    ask = _Ask(farm)
    assert (await ask(BLACK))[0] in {1, 3}
    assert (await ask(BLACK))[0] in {1, 3}

    printer_id, reason = await ask(WHITE)
    assert printer_id == 2, f"the white job must still find the AMS printer, got {printer_id} ({reason})"
