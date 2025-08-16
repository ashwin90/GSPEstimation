### Description

Here we provide code associated with the paper *The Generalized Stochastic Preference Choice Model* (Berbeglia and Venkataraman, 2025).
The code was tested with the dependencies provided in the `requirements.txt` file and Python version 3.10.9

## Usage
The data format is in json. We have provided two example instances `instance_1.dt` and `instance_2.dt`. The main file is `estimate.py` which estimates a GSP model on the *in_sample* transactions in the input instance, and then prints the predicted choice probabilities for the *out_of_sample* transactions in the input instance. The code can be run as follows
```
python estimate.py <input_instance> <max_choice_index> <ub_proportion>
```
where

- `input_instance`: input filename--e.g., instance_1.dt
- `max_choice_index`: maximum choice index for non-standard type ($k_{\max}$ in Section 4.1)---e.g., 3
- `ub_proportion`: upper bound on total proportion of non-standard types ($\delta$ in Section 4.1)---e.g., 0.5