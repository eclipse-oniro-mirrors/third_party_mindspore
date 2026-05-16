SCRIPT_DIR="$(cd "$(dirname "{BASH_SOURCE[0]}")" && pwd)"
pip install -r $SCRIPT_DIR/requirements.txt
pip install onnxslim==0.1.73 --no-deps