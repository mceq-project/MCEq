Link the macOS sparse extension explicitly against Accelerate. Give the
docs dependency group its Python 3.11 minimum so `uv sync` can resolve it
without raising MCEq's Python 3.10 runtime minimum.
