# Sacramento Model Verification Report

**Date**: 2026-01-19 17:33:16
**Python Version**: 3.12.8

## Summary

| Test Case | Status | Max Difference | Variable |
|-----------|--------|----------------|----------|
| TC01: Default parameters, full dataset | PASS | 2.84e-14 | lztwc |
| TC02: Dry catchment scenario | PASS | 1.42e-14 | lztwc |
| TC03: Wet catchment scenario | PASS | 2.84e-14 | lztwc |
| TC04: High impervious area | PASS | 2.84e-14 | lztwc |
| TC05: Deep groundwater | PASS | 2.84e-14 | lztwc |
| TC06: Unit hydrograph lag | PASS | 2.84e-14 | lztwc |
| TC07: Zero rainfall (evap only) | PASS | 0.00e+00 | N/A |
| TC08: Storm event (100mm pulse) | PASS | 7.11e-15 | lztwc |
| TC09: Stores initialized full | PASS | 2.84e-14 | lztwc |
| TC10: Long dry spell | PASS | 0.00e+00 | N/A |

## Overall Result

**ALL TESTS PASSED**

## Detailed Results

### TC01: Default parameters, full dataset

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 3.55e-15 | 5.44e-17 | 658 |
| baseflow | 2.22e-16 | 4.73e-17 | 259 |
| uztwc | 7.11e-15 | 3.46e-16 | 9 |
| uzfwc | 3.55e-15 | 6.62e-17 | 56 |
| lztwc | 2.84e-14 | 1.29e-15 | 298 |
| lzfsc | 3.55e-15 | 2.31e-16 | 319 |
| lzfpc | 7.11e-15 | 4.90e-16 | 254 |
| channel_flow | 3.55e-15 | 5.44e-17 | 658 |
| evap_uztw | 4.44e-16 | 4.27e-17 | 32 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 2.22e-16 | 3.45e-17 | 425 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 6.31e-30 | 3.43e-31 | 595 |

### TC02: Dry catchment scenario

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 7.11e-15 | 5.67e-17 | 792 |
| baseflow | 2.22e-16 | 4.38e-17 | 966 |
| uztwc | 3.55e-15 | 4.04e-16 | 5 |
| uzfwc | 1.78e-15 | 8.41e-17 | 56 |
| lztwc | 1.42e-14 | 3.75e-16 | 234 |
| lzfsc | 3.55e-15 | 2.60e-16 | 320 |
| lzfpc | 7.11e-15 | 3.41e-16 | 314 |
| channel_flow | 7.11e-15 | 5.67e-17 | 792 |
| evap_uztw | 4.44e-16 | 4.35e-17 | 50 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 2.22e-16 | 3.88e-17 | 70 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 6.31e-30 | 2.32e-31 | 628 |

### TC03: Wet catchment scenario

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 3.55e-15 | 5.44e-17 | 629 |
| baseflow | 4.44e-16 | 4.55e-17 | 310 |
| uztwc | 1.42e-14 | 2.21e-15 | 109 |
| uzfwc | 3.55e-15 | 3.91e-17 | 937 |
| lztwc | 2.84e-14 | 2.10e-15 | 206 |
| lzfsc | 3.55e-15 | 1.55e-16 | 316 |
| lzfpc | 1.42e-14 | 5.29e-16 | 322 |
| channel_flow | 3.55e-15 | 5.44e-17 | 629 |
| evap_uztw | 4.44e-16 | 4.99e-17 | 43 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 2.22e-16 | 3.55e-17 | 433 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 1.26e-29 | 3.87e-31 | 346 |

### TC04: High impervious area

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 3.55e-15 | 6.79e-17 | 792 |
| baseflow | 2.22e-16 | 4.58e-17 | 308 |
| uztwc | 7.11e-15 | 3.46e-16 | 9 |
| uzfwc | 3.55e-15 | 6.62e-17 | 56 |
| lztwc | 2.84e-14 | 1.29e-15 | 298 |
| lzfsc | 3.55e-15 | 2.31e-16 | 319 |
| lzfpc | 7.11e-15 | 4.90e-16 | 254 |
| channel_flow | 3.55e-15 | 6.79e-17 | 792 |
| evap_uztw | 4.44e-16 | 4.67e-17 | 39 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 2.22e-16 | 3.76e-17 | 423 |
| e5 | 1.11e-16 | 4.79e-17 | 11 |
| mass_balance | 4.44e-16 | 4.33e-17 | 446 |

### TC05: Deep groundwater

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 4.44e-16 | 4.20e-17 | 323 |
| baseflow | 4.44e-16 | 4.30e-17 | 323 |
| uztwc | 7.11e-15 | 3.46e-16 | 9 |
| uzfwc | 2.22e-16 | 1.24e-18 | 682 |
| lztwc | 2.84e-14 | 1.43e-15 | 230 |
| lzfsc | 7.11e-15 | 2.94e-16 | 364 |
| lzfpc | 2.84e-14 | 2.32e-15 | 358 |
| channel_flow | 4.44e-16 | 4.20e-17 | 323 |
| evap_uztw | 4.44e-16 | 4.27e-17 | 32 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 2.22e-16 | 3.53e-17 | 81 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 7.11e-15 | 4.50e-16 | 312 |

### TC06: Unit hydrograph lag

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 7.11e-15 | 2.38e-16 | 346 |
| baseflow | 4.44e-16 | 4.97e-17 | 374 |
| uztwc | 7.11e-15 | 3.46e-16 | 9 |
| uzfwc | 3.55e-15 | 6.62e-17 | 56 |
| lztwc | 2.84e-14 | 1.29e-15 | 298 |
| lzfsc | 3.55e-15 | 2.31e-16 | 319 |
| lzfpc | 7.11e-15 | 4.90e-16 | 254 |
| channel_flow | 7.11e-15 | 2.38e-16 | 346 |
| evap_uztw | 4.44e-16 | 4.27e-17 | 32 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 2.22e-16 | 3.45e-17 | 425 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 1.42e-14 | 1.67e-16 | 1031 |

### TC07: Zero rainfall (evap only)

- Input records: 365
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 0.00e+00 | 0.00e+00 | 0 |
| baseflow | 0.00e+00 | 0.00e+00 | 0 |
| uztwc | 0.00e+00 | 0.00e+00 | 0 |
| uzfwc | 0.00e+00 | 0.00e+00 | 0 |
| lztwc | 0.00e+00 | 0.00e+00 | 0 |
| lzfsc | 0.00e+00 | 0.00e+00 | 0 |
| lzfpc | 0.00e+00 | 0.00e+00 | 0 |
| channel_flow | 0.00e+00 | 0.00e+00 | 0 |
| evap_uztw | 0.00e+00 | 0.00e+00 | 0 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 0.00e+00 | 0.00e+00 | 0 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 0.00e+00 | 0.00e+00 | 0 |

### TC08: Storm event (100mm pulse)

- Input records: 91
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 1.01e-16 | 2.85e-17 | 49 |
| baseflow | 1.01e-16 | 3.02e-17 | 49 |
| uztwc | 1.78e-15 | 7.08e-17 | 50 |
| uzfwc | 1.78e-15 | 1.95e-17 | 30 |
| lztwc | 7.11e-15 | 3.12e-16 | 32 |
| lzfsc | 1.11e-16 | 3.26e-17 | 30 |
| lzfpc | 2.22e-16 | 2.93e-17 | 42 |
| channel_flow | 1.01e-16 | 2.85e-17 | 49 |
| evap_uztw | 4.44e-16 | 3.39e-17 | 31 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 1.11e-16 | 2.99e-17 | 40 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 7.89e-31 | 2.82e-32 | 35 |

### TC09: Stores initialized full

- Input records: 1095
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 3.55e-15 | 7.80e-17 | 792 |
| baseflow | 2.22e-16 | 4.50e-17 | 3 |
| uztwc | 7.11e-15 | 3.36e-16 | 7 |
| uzfwc | 3.55e-15 | 8.56e-17 | 0 |
| lztwc | 2.84e-14 | 1.70e-15 | 300 |
| lzfsc | 3.55e-15 | 2.19e-16 | 12 |
| lzfpc | 7.11e-15 | 5.27e-16 | 10 |
| channel_flow | 3.55e-15 | 7.80e-17 | 792 |
| evap_uztw | 4.44e-16 | 4.21e-17 | 47 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 4.44e-16 | 3.87e-17 | 466 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 3.16e-30 | 3.42e-31 | 203 |

### TC10: Long dry spell

- Input records: 180
- C# reference available: True
- Status: PASS

**Variable Statistics:**

| Variable | Max Difference | Mean Difference | Timestep of Max |
|----------|----------------|-----------------|-----------------|
| runoff | 0.00e+00 | 0.00e+00 | 0 |
| baseflow | 0.00e+00 | 0.00e+00 | 0 |
| uztwc | 0.00e+00 | 0.00e+00 | 0 |
| uzfwc | 0.00e+00 | 0.00e+00 | 0 |
| lztwc | 0.00e+00 | 0.00e+00 | 0 |
| lzfsc | 0.00e+00 | 0.00e+00 | 0 |
| lzfpc | 0.00e+00 | 0.00e+00 | 0 |
| channel_flow | 0.00e+00 | 0.00e+00 | 0 |
| evap_uztw | 0.00e+00 | 0.00e+00 | 0 |
| evap_uzfw | 0.00e+00 | 0.00e+00 | 0 |
| e3 | 0.00e+00 | 0.00e+00 | 0 |
| e5 | 0.00e+00 | 0.00e+00 | 0 |
| mass_balance | 0.00e+00 | 0.00e+00 | 0 |
