import logging
import numpy as np
from openvino.runtime import Core, LogLevel  # Correctly import LogLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Initialize OpenVINO Runtime
core = Core()

# Set CPU plugin to DEBUG level to capture detailed logs
core.set_property("CPU", {"LOG_LEVEL": LogLevel.DEBUG})  # Use LogLevel.DEBUG directly

# Load the model
model_path = "path_to_your_model.xml"  # Replace with your actual model path
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model=model, device_name="CPU")

# Prepare input data
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)

# Run inference
result = compiled_model.infer_new_request({compiled_model.inputs[0]: input_tensor})

# Process the result as needed
print("Inference completed.")

#check if it is AMX-enebled in the log.
