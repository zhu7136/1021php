"""Run the complete established resume suite against276 without changing old guards."""
from pathlib import Path
source=Path(__file__).with_name('test_g1_resume224.py').read_text()
source=(source.replace('224','276').replace('841','1718').replace('842','1720')
        .replace('16840','34400').replace("863)","1741)").replace("861)","1738)"))
exec(compile(source,str(Path(__file__).resolve()),'exec'),globals())