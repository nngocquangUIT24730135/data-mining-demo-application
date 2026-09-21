from __future__ import annotations

import sys


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    from datamining_app.ui.app import MainApp

    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
