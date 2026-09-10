Pass the ETD2 driver's row-major state directly to Apple Accelerate SpMM,
removing transposing copies and their scratch buffers. Construct native
matrices from CSR rows to lower peak memory, and release construction
arrays after commit. Validate dense layouts, precisions, ownership,
padded strides, and coupled 2D Accelerate solves against NumPy.
