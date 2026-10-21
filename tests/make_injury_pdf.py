"""Build a test PDF laid out like the NBA's official injury report.

Page size, column positions and the title/footer text copy the real
2023-26 layout (column edges measured from real reports). It covers the
cases the parser has to handle: values printed once per group, a reason
that wraps onto two lines (which pushes the other cells between them),
'NOT YET SUBMITTED', and a second page with the header repeated.
The players and teams here are for tests only.
"""
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas

COLS = [22, 112, 187, 259, 376, 547, 659]   # left edge of each column
ROW = 13.0


def build(path, rows_page1, rows_page2=(), title="Injury Report: 10/21/26 05:30 PM"):
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    w, h = landscape(A4)
    pages = [rows_page1] + ([rows_page2] if rows_page2 else [])
    for n, rows in enumerate(pages, start=1):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(330, h - 40, title)
        c.setFont("Helvetica-Bold", 8)
        for x, head in zip(COLS, ["Game Date", "Game Time", "Matchup", "Team", "Player Name",
                                  "Current Status", "Reason"]):
            c.drawString(x, h - 70, head)
        c.setFont("Helvetica", 8)
        y = h - 86
        for r in rows:
            reason = r[6] if isinstance(r[6], (list, tuple)) else [r[6]]
            if len(reason) == 2:
                # wrapped reason: other cells sit vertically between its two lines
                c.drawString(COLS[6], y, reason[0])
                mid = y - ROW / 2
                for x, v in zip(COLS[:6], r[:6]):
                    if v:
                        c.drawString(x, mid, v)
                c.drawString(COLS[6], y - ROW, reason[1])
                y -= 2 * ROW
            else:
                for x, v in zip(COLS, list(r[:6]) + [reason[0]]):
                    if v:
                        c.drawString(x, y, v)
                y -= ROW
        c.drawString(390, 25, f"Page {n} of {len(pages)}")
        c.showPage()
    c.save()


SAMPLE_P1 = [
    ("10/21/2026", "07:00 (ET)", "BOS@DET", "Boston Celtics", "Tester, Alpha", "Out",
     "Injury/Illness - Left Knee; Surgery"),
    ("", "", "", "", "Sample Jr., Bravo", "Questionable", "Injury/Illness - Right Ankle; Sprain"),
    ("", "", "", "Detroit Pistons", "Example, Charlie", "Doubtful",
     ("Injury/Illness - Left Posterior Tibialis", "Tendon; Surgery Rehabilitation")),
    ("", "", "", "", "Placeholder, Delta", "Out", "G League - Two-Way"),
    ("", "07:30 (ET)", "PHI@NYK", "Philadelphia 76ers", "NOT YET SUBMITTED", "", ""),
    ("", "", "", "New York Knicks", "Dummy, Echo", "Probable", "Injury/Illness - Low Back; Soreness"),
]
SAMPLE_P2 = [
    ("10/22/2026", "10:00 (ET)", "OKC@LAL", "Oklahoma City Thunder", "Mock, Foxtrot", "Available",
     "Injury/Illness - Right Hand; Contusion"),
    ("", "", "", "Los Angeles Lakers", "Tryout, Golf", "Questionable", "Rest"),
]
