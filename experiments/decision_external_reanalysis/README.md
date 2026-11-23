# Decision consequence and corrected Davis re-selection

This bundle addresses two evidence-chain gaps without rerunning docking.

1. It measures the paired chemical consequences of the q25 learning-label floor on the 21 strict historical tasks, using the original evaluation labels.
2. It retrains the global model from the 21 strict tasks with cross-fitted donor calibration features and re-selects the fixed nine Davis tasks without exposing their evaluation labels to selection.
3. It evaluates Davis selections both as plug-in point values and as intervals induced by the documented 10,000 nM reporting ceiling.

The Davis analysis is retrospective because the labels have been examined previously. The label firewall is executable and checks that arbitrary changes to evaluation labels cannot alter a selection.

Run the resource audit before either analysis:

```bash
python runner.py audit
python runner.py all
```
