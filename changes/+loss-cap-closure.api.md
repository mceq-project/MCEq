The continuous-loss step cap (`continuous_loss_rate`) ignores the upwind
closure rows of a loss band and counts only the mixed-sign rows above them
(`MatrixBuilder._contloss_upwind_rows`). For the 2D FLUKA configuration the cap
rises from 0.185 g/cm^2 to above `dX_max`, so `dX_max` sets the step; set it
explicitly to recover the smaller step.
