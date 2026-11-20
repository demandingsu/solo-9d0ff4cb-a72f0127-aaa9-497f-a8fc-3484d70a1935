"""SQLite access for the lake temperature archive.

Design invariant
----------------
The restored temperature (``restored_mc``) of every frame is persisted and is
authoritative.  A frame's *stored* representation is derived from it:

    absolute frame : stored = restored
    relative frame : stored = restored - restored(base_seq)

Consequently, correcting a base frame

  * changes the restored temperature of that frame only;
  * leaves every other frame's restored temperature and base reference intact;
  * only changes the derived stored difference of frames that reference it.

A correction reads and writes just the addressed frame (plus, for a relative
frame, its base's restored value).

Stale-page protection
---------------------
A correction carries ``seen_restored_mc``: the temperature the page displayed.
If the archive's current restored value differs, the submission is rejected
with 409 and nothing is written.  A submission whose restored value is
unchanged is accepted even when the stored representation has drifted (e.g.
Be aware that this protects the single-frame correction page; a supplement
upload is a separate, batch path described below.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from . import csvio

_SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


class ArchiveError(Exception):
    """Domain error carrying an HTTP status code for the UI."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


class ConflictError(ArchiveError):
    def __init__(self, message: str):
        super().__init__(message, status=409)


class SupplementError(ArchiveError):
    """A supplement upload that cannot be accepted; nothing was written."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message, status=status)


@dataclass(frozen=True)
class FrameView:
    seq: int
    sampled_at: str
    sampled_at_raw: str
    restored_mc: int
    base_seq: int | None
    stored_kind: str
    stored_mc: int
    kind_raw: str = ""
    extras: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchiveSummary:
    id: int
    name: str
    source_filename: str
    frame_count: int
    created_at: str


@dataclass(frozen=True)
class SupplementResult:
    added: tuple[int, ...]
    skipped: tuple[int, ...]


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as fh:
            conn.executescript(fh.read())
        _migrate(conn)
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring a database created by an older release up to this schema.

    Archives imported before ``initial_restored_mc`` existed have no manual
    corrections on record, so their current restored value is the value they
    were first archived with.  They remain directly supplantable.
    """
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(frame)")}
    if "initial_restored_mc" not in columns:
        conn.execute("ALTER TABLE frame ADD COLUMN initial_restored_mc INTEGER")
        conn.execute("UPDATE frame SET initial_restored_mc = restored_mc"
                     " WHERE initial_restored_mc IS NULL")


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def _stored(restored_mc: int, base_restored_mc: int | None) -> tuple[str, int]:
    if base_restored_mc is None:
        return "absolute", restored_mc
    return "delta", restored_mc - base_restored_mc


def _view(row: sqlite3.Row, base_restored_mc: int | None) -> FrameView:
    kind, stored = _stored(row["restored_mc"], base_restored_mc)
    return FrameView(
        seq=row["seq"],
        sampled_at=row["sampled_at"],
        sampled_at_raw=row["sampled_at_raw"],
        restored_mc=row["restored_mc"],
        base_seq=row["base_seq"],
        stored_kind=kind,
        stored_mc=stored,
        kind_raw=row["kind_raw"],
        extras=tuple(json.loads(row["extras_json"])) if row["extras_json"] else (),
    )


def _fetch_base_restored(conn, archive_id: int, base_seq: int | None) -> int | None:
    if base_seq is None:
        return None
    row = conn.execute(
        "SELECT restored_mc FROM frame WHERE archive_id = ? AND seq = ?",
        (archive_id, base_seq),
    ).fetchone()
    if row is None:
        raise ArchiveError("base frame %d is missing from the archive" % base_seq, status=500)
    return row["restored_mc"]


def create_archive(db_path: str, name: str, filename: str, payload: bytes) -> int:
    """Import a logger CSV as a new archive; returns the archive id."""
    rows, layout = csvio.parse_csv(payload)
    restored: dict[int, int] = {}
    for fr in rows:  # ordered by seq, and base_seq < seq
        if fr.stored_kind == "absolute":
            restored[fr.seq] = fr.stored_mc
        else:
            restored[fr.seq] = fr.stored_mc + restored[fr.base_seq]

    conn = connect(db_path)
    try:
        with _transaction(conn):
            try:
                cur = conn.execute(
                    "INSERT INTO archive (name, source_filename, layout_json)"
                    " VALUES (?, ?, ?)",
                    (name, filename, layout.to_json()),
                )
            except sqlite3.IntegrityError as exc:
                if "archive.name" in str(exc):
                    raise ConflictError(
                        "an archive named %r already exists" % name) from exc
                raise
            archive_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO frame (archive_id, seq, sampled_at, sampled_at_raw,"
                " restored_mc, initial_restored_mc, base_seq, kind_raw, extras_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(archive_id, fr.seq, fr.sampled_at, fr.sampled_at_raw,
                  restored[fr.seq], restored[fr.seq], fr.base_seq, fr.kind_raw,
                  json.dumps(list(fr.extras), ensure_ascii=False)) for fr in rows],
            )
        return archive_id
    finally:
        conn.close()


def list_archives(db_path: str) -> list[ArchiveSummary]:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT a.id, a.name, a.source_filename, a.created_at, COUNT(f.seq) AS n"
            " FROM archive a LEFT JOIN frame f ON f.archive_id = a.id"
            " GROUP BY a.id ORDER BY a.id"
        ).fetchall()
        return [
            ArchiveSummary(id=r["id"], name=r["name"],
                           source_filename=r["source_filename"],
                           frame_count=r["n"], created_at=r["created_at"])
            for r in rows
        ]
    finally:
        conn.close()


def _archive_row(conn, archive_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM archive WHERE id = ?", (archive_id,)).fetchone()
    if row is None:
        raise ArchiveError("archive not found", status=404)
    return row


def get_archive(db_path: str, archive_id: int) -> ArchiveSummary:
    conn = connect(db_path)
    try:
        row = _archive_row(conn, archive_id)
        n = conn.execute(
            "SELECT COUNT(*) FROM frame WHERE archive_id = ?", (archive_id,)
        ).fetchone()[0]
        return ArchiveSummary(id=row["id"], name=row["name"],
                              source_filename=row["source_filename"],
                              frame_count=n, created_at=row["created_at"])
    finally:
        conn.close()


def get_frame(dbp: str, archive_id: int, seq: int) -> FrameView:
    """Read just the addressed frame (and its base's restored value).

    Only rows of this archive are read.
    """
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        row = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? AND seq = ?", (archive_id, seq)
        ).fetchone()
        if row is None:
            raise ArchiveError("frame %d not found in this archive" % seq, status=404)
        return _view(row, _fetch_base_restored(conn, archive_id, row["base_seq"]))
    finally:
        conn.close()


def get_frame_bundle(dbp: str, archive_id: int, seq: int):
    """Read the addressed frame, its base, and the frames that cite it as base.

    Used only to render the single-frame page.  The correction path itself
    reads and writes nothing but the addressed frame (and the base value it is
    relative to); this listing merely warns the operator which stored
    differences will move when a base frame is corrected.
    """
    conn = connect(dbp)
    try:
        _archive_row(conn, archive_id)
        row = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? AND seq = ?", (archive_id, seq)
        ).fetchone()
        if row is None:
            raise ArchiveError("frame %d not found in this archive" % seq, status=404)
        base_val = _fetch_base_restored(conn, archive_id, row["base_seq"])
        base = None
        if row["base_seq"] is not None:
            base_row = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? AND seq = ?",
                (archive_id, row["base_seq"]),
            ).fetchone()
            base = _view(base_row, _fetch_base_restored(conn, archive_id, base_row["base_seq"]))

        ref_rows = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? AND base_seq = ? ORDER BY seq",
            (archive_id, seq),
        ).fetchall()
        referrers = [_view(r, row["restored_mc"]) for r in ref_rows]
        return _view(row, base_val), base, referrers
    finally:
        conn.close()


def list_frames(db_path: str, archive_id: int) -> list[FrameView]:
    conn = connect(db_path)
    try:
        _archive_row(conn, archive_id)
        rows = conn.execute(
            "SELECT * FROM frame WHERE archive_id = ? ORDER BY seq", (archive_id,)
        ).fetchall()
        by_seq = {r["seq"]: r["restored_mc"] for r in rows}
        return [
            _view(r, by_seq.get(r["base_seq"]) if r["base_seq"] is not None else None)
            for r in rows
        ]
    finally:
        conn.close()


def correct_frame(db_path: str, archive_id: int, seq: int, *,
                  seen_restored_mc: int, new_restored_mc: int) -> FrameView:
    """Apply a correction to one frame; return the updated view.

    Raises :class:`ConflictError` when the temperature the submitter saw no
    longer matches the stored restored value.  Nothing is written in that case.
    """
    if new_restored_mc < -1_000_000 or new_restored_mc > 1_000_000:
        raise ArchiveError("restored temperature is outside the supported range")

    conn = connect(db_path)
    try:
        with _transaction(conn):
            _archive_row(conn, archive_id)
            row = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? AND seq = ?",
                (archive_id, seq),
            ).fetchone()
            if row is None:
                raise ArchiveError("frame %d not found in this archive" % seq, status=404)

            old_restored = row["restored_mc"]
            old_base = row["base_seq"]
            old_base_restored = _fetch_base_restored(conn, archive_id, old_base)
            old_kind, old_stored = _stored(old_restored, old_base_restored)

            if seen_restored_mc != old_restored:
                raise ConflictError(
                    "页面显示的温度是 %d m°C，但该帧当前的还原值是 %d m°C；"
                    "更正已拒绝，现档保持不变。" % (seen_restored_mc, old_restored)
                )

            new_kind, new_stored = _stored(new_restored_mc, old_base_restored)

            if new_restored_mc != old_restored:
                conn.execute(
                    "UPDATE frame SET restored_mc = ? WHERE archive_id = ? AND seq = ?",
                    (new_restored_mc, archive_id, seq),
                )
                conn.execute(
                    "INSERT INTO revision (archive_id, seq, old_restored_mc, old_base_seq,"
                    " old_stored_kind, old_stored_mc, new_restored_mc, new_base_seq,"
                    " new_stored_kind, new_stored_mc, seen_restored_mc)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (archive_id, seq, old_restored, old_base, old_kind, old_stored,
                     new_restored_mc, old_base, new_kind, new_stored, seen_restored_mc),
                )

            return FrameView(
                seq=row["seq"], sampled_at=row["sampled_at"],
                sampled_at_raw=row["sampled_at_raw"],
                restored_mc=new_restored_mc, base_seq=old_base,
                stored_kind=new_kind, stored_mc=new_stored,
                kind_raw=row["kind_raw"],
                extras=tuple(json.loads(row["extras_json"])),
            )
    finally:
        conn.close()


def _layout_fingerprint(layout: csvio.CsvLayout) -> tuple:
    """What "same header and column order" means for a supplement upload."""
    return (layout.header, layout.delimiter, layout.col_seq, layout.col_time,
            layout.col_value, layout.col_base, layout.col_kind, layout.extra_cols)


def _original_restored(file_frames: dict[int, int],
                       archive_initial: dict[int, int], base_seq: int) -> int | None:
    """First-archived restored temperature of ``base_seq``, or None if unknown.

    A base within the uploaded file is decoded from the recorder's own values;
    a base already in the archive contributes the restored temperature it was
    first archived with, never a later manual correction.
    """
    if base_seq in archive_initial:
        return archive_initial[base_seq]
    return file_frames.get(base_seq)


def supplement_archive(db_path: str, archive_id: int, filename: str,
                       payload: bytes) -> SupplementResult:
    """Merge a recorder's re-transmitted CSV into an existing archive.

    The uploaded file must carry the same header and column order as the
    archive's original file.  A frame is decoded from the recorder's *original*
    values: an absolute frame from its own value; a delta frame from its delta
    plus the first-archived restored temperature of its base -- whether that
    base is earlier in the same file or already in the archive.  Manual
    corrections already made in the archive therefore never leak into the
    decode.

    Frame numbers already in the archive keep their archived state.  Such a
    frame is skipped when both its sampling instant and its original restored
    temperature match the archive (so manual corrections survive); any other
    difference is a conflict, and the whole file is rejected naming the frame
    numbers, with the archive untouched.

    Only the uploaded file's own frames and the bases they cite are examined.
    The checks and the insert run in one ``BEGIN IMMEDIATE`` transaction, so
    two supplements submitted at the same time are applied one after the other
    under exactly these rules.
    """
    rows, layout = csvio.parse_csv(payload, require_bases_in_file=False)
    if not rows:
        raise SupplementError("文件不含任何帧。", status=400)

    conn = connect(db_path)
    try:
        with _transaction(conn):
            archive_row = _archive_row(conn, archive_id)
            stored_layout = csvio.CsvLayout.from_json(archive_row["layout_json"])
            if _layout_fingerprint(layout) != _layout_fingerprint(stored_layout):
                raise SupplementError(
                    "文件表头或列序与原档不一致：原档为 %s，本次为 %s。"
                    "整份退回，档案保持原样。"
                    % (_describe_layout(stored_layout), _describe_layout(layout)),
                    status=400)

            # Only the frames this file carries, and the bases they cite, are
            # looked at; the rest of the archive is not examined.
            touched = sorted({fr.seq for fr in rows}
                             | {fr.base_seq for fr in rows
                                if fr.base_seq is not None})
            marks = ",".join("?" * len(touched))
            archive_initial: dict[int, int] = {}
            archive_sampled: dict[int, str] = {}
            for r in conn.execute(
                    "SELECT seq, sampled_at, initial_restored_mc FROM frame"
                    " WHERE archive_id = ? AND seq IN (%s)" % marks,
                    (archive_id, *touched)).fetchall():
                archive_initial[r["seq"]] = r["initial_restored_mc"]
                archive_sampled[r["seq"]] = r["sampled_at"]

            # Decode the uploaded frames from the recorder's own values.  An
            # absolute frame is its value; a delta frame is its delta plus the
            # base's first-archived restored temperature, with bases earlier in
            # this file decoded first (rows are ordered by seq).
            decoded: dict[int, int] = {}
            no_base: list[tuple[int, int]] = []
            for fr in rows:
                if fr.stored_kind == "absolute":
                    decoded[fr.seq] = fr.stored_mc
                    continue
                base_original = _original_restored(
                    decoded, archive_initial, fr.base_seq)
                if base_original is None:
                    no_base.append((fr.seq, fr.base_seq))
                    continue
                decoded[fr.seq] = fr.stored_mc + base_original
            if no_base:
                raise SupplementError(
                    "缺少基准帧，整份退回，档案保持原样。相关帧：%s。"
                    % "；".join("帧 %d（缺基准帧 %d）" % pair for pair in no_base))

            # A frame number already on file must agree with the archive; an
            # identical one is skipped so any manual correction is kept.
            conflicts: list[tuple[int, str]] = []
            skipped: list[int] = []
            for fr in rows:
                if fr.seq not in archive_initial:
                    continue
                if fr.sampled_at != archive_sampled[fr.seq]:
                    conflicts.append((fr.seq, "采样时刻不同"))
                elif decoded[fr.seq] != archive_initial[fr.seq]:
                    conflicts.append((fr.seq, "原始还原温度不同"))
                else:
                    skipped.append(fr.seq)
            if conflicts:
                raise SupplementError(
                    "补传与档案不一致，整份退回，档案保持原样。相关帧：%s。"
                    % "；".join("帧 %d（%s）" % pair for pair in conflicts))

            new_rows = [fr for fr in rows if fr.seq not in archive_initial]
            conn.executemany(
                "INSERT INTO frame (archive_id, seq, sampled_at, sampled_at_raw,"
                " restored_mc, initial_restored_mc, base_seq, kind_raw, extras_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(archive_id, fr.seq, fr.sampled_at, fr.sampled_at_raw,
                  decoded[fr.seq], decoded[fr.seq], fr.base_seq, fr.kind_raw,
                  json.dumps(list(fr.extras), ensure_ascii=False))
                 for fr in new_rows],
            )
            if new_rows:
                conn.execute(
                    "UPDATE archive SET source_filename = ? WHERE id = ?",
                    (filename, archive_id))
            conn.execute(
                "INSERT INTO supplement (archive_id, source_filename, added_seqs,"
                " skipped_seqs) VALUES (?, ?, ?, ?)",
                (archive_id, filename,
                 json.dumps([fr.seq for fr in new_rows]),
                 json.dumps(sorted(skipped))),
            )
        return SupplementResult(added=tuple(fr.seq for fr in new_rows),
                                skipped=tuple(sorted(skipped)))
    finally:
        conn.close()


def _describe_layout(layout: csvio.CsvLayout) -> str:
    text = layout.delimiter.join(layout.header)
    if layout.delimiter != ",":
        text += "（分隔符 %r）" % layout.delimiter
    return repr(text)


def export_csv(db_path: str, archive_id: int) -> bytes:
    """Render every frame of one archive from a single consistent snapshot.

    The whole read runs inside one transaction, so the file reflects one
    committed state; a correction cannot be half-visible in a download.
    """
    conn = connect(db_path)
    try:
        conn.execute("BEGIN")
        try:
            row = _archive_row(conn, archive_id)
            layout = csvio.CsvLayout.from_json(row["layout_json"])
            rows = conn.execute(
                "SELECT * FROM frame WHERE archive_id = ? ORDER BY seq", (archive_id,)
            ).fetchall()
            by_seq = {r["seq"]: r["restored_mc"] for r in rows}
            frames = [
                _view(r, by_seq.get(r["base_seq"]) if r["base_seq"] is not None else None)
                for r in rows
            ]
        finally:
            conn.execute("COMMIT")
    finally:
        conn.close()
    return csvio.render_csv(frames, layout)
