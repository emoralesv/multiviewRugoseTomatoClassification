from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent
src_dir = project_root / "databaseMLUtils" / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
elif not src_dir.exists():
    raise ModuleNotFoundError(f"Expected databaseMLUtils sources at {src_dir}")

from PIL import Image
import matplotlib.pyplot as plt

from databaseMLUtils.transforms import Transformer
from databaseMLUtils.converter import convert_xml_to_Classification


transforms = Transformer()
transforms.print()



from databaseMLUtils.reporting import make_dataset_report


out_dir  = r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\report"
make_dataset_report(
    data=r"I:\Documentos\Documentos\repos\multiviewRugoseTomatoClassification\views\RGB",
    name="Tomate Rugoso Clasificaciónn",
    url=out_dir,
    out="report",
)




