Compact secant coupling template: `config.secant_compact_coupling = True`
applies the mode coupling through column-restricted operators
(`F(x + scatter(YG)) = F(x) + off[:, idx] @ YG`), removing the full-state
`w` copy and the fancy-index scatter from every step stage on all three
backends (numpy, MKL, CUDA) at the price of one compact SpMM. Same
arithmetic reordered; default off preserves bit-identity.
