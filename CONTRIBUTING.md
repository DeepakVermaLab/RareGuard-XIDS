# Contributing

RareGuard-XIDS is a research implementation with a frozen evidence protocol. Contributions are welcome when they preserve reproducibility and claim integrity.

## Before opening a change

1. Create a focused branch.
2. Keep dataset paths and credentials outside Git.
3. Add or update tests for implementation changes.
4. Record protocol changes separately from result changes.
5. Never manually modify metrics, ablations or result tables to improve a claim.

## Validation

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
pytest -q
```

New experiments must write structured logs, retain seed-level rows and preserve validation/test separation.
