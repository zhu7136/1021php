"""Private-task entry; starts one bounded screen and never submits another task."""
from five_skill_worker import main

if __name__=='__main__':
    raise SystemExit(main())