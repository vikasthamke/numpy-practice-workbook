"""
practice_system.py
-------------------
Interactive hint/answer engine for numpy_practice.ipynb.

Usage inside the notebook:
    from practice_system import load_dataset, hint, answer
    temperature, vibration, pressure, production, downtime, machine_ids, n_machines, n_hours = load_dataset()
    hint(12)
    hint(20, level=2)
    answer(12)

This file intentionally lives outside the notebook so solutions aren't
visible until you ask for them. It's a convenience separation, not
encryption -- you're welcome to read this file directly, but you'll get
more out of the workbook if you attempt each task first.

answer(n) doesn't just print a solution's code -- it actually runs that
code against a freshly-generated copy of the dataset and shows you the
real output, so you can check your own result against it directly instead
of just reading code and guessing what it produces.
"""

import contextlib
import io
import os
import tempfile

import numpy as np

_MAX_OUTPUT_CHARS = 2500


def _build_dataset_dict():
    """Deterministically (re)generates the Smart Factory sensor dataset. Saves and restores
    NumPy's global random state so calling this never disturbs whatever the caller (the
    notebook, or a task's own solution code) was doing with randomness."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(42)

        n_machines = 12
        n_hours = 120  # 5 days x 24 hours

        temperature = np.random.normal(loc=65, scale=4, size=(n_machines, n_hours)).round(2)
        vibration = np.random.normal(loc=2.5, scale=0.6, size=(n_machines, n_hours)).round(2)
        pressure = np.random.normal(loc=101, scale=3, size=(n_machines, n_hours)).round(2)

        # Sensor fault spikes
        temp_spike_rows = np.random.randint(0, n_machines, 15)
        temp_spike_cols = np.random.randint(0, n_hours, 15)
        temperature[temp_spike_rows, temp_spike_cols] += np.random.uniform(25, 45, 15)

        vib_spike_rows = np.random.randint(0, n_machines, 15)
        vib_spike_cols = np.random.randint(0, n_hours, 15)
        vibration[vib_spike_rows, vib_spike_cols] += np.random.uniform(3, 6, 15)

        # Missing temperature readings (sensor dropout)
        nan_rows = np.random.randint(0, n_machines, 10)
        nan_cols = np.random.randint(0, n_hours, 10)
        temperature[nan_rows, nan_cols] = np.nan

        # Zero-pressure wiring faults
        zero_rows = np.random.randint(0, n_machines, 6)
        zero_cols = np.random.randint(0, n_hours, 6)
        pressure[zero_rows, zero_cols] = 0

        # Daily production output per machine (units) across 5 days
        production = np.random.randint(180, 260, size=(n_machines, 5))

        # Total downtime hours per machine over the period
        downtime = np.random.uniform(0, 12, size=n_machines).round(1)

        machine_ids = [f"M{str(i+1).zfill(2)}" for i in range(n_machines)]
    finally:
        np.random.set_state(rng_state)

    return {
        "temperature": temperature, "vibration": vibration, "pressure": pressure,
        "production": production, "downtime": downtime, "machine_ids": machine_ids,
        "n_machines": n_machines, "n_hours": n_hours,
    }


def load_dataset():
    """Public entry point the notebook's setup cell calls. Returns a fresh, independent copy
    of the dataset every time -- unpack it into your own variables:

        temperature, vibration, pressure, production, downtime, machine_ids, n_machines, n_hours = load_dataset()
    """
    d = _build_dataset_dict()
    return (d["temperature"], d["vibration"], d["pressure"], d["production"],
            d["downtime"], d["machine_ids"], d["n_machines"], d["n_hours"])


def _run_solution(code_str):
    """Execute a solution's code against its own freshly-generated copy of the dataset and
    capture whatever it prints. Runs inside a throwaway temp directory (so a task like the
    save/load one doesn't leave files in your working directory) and saves/restores the
    global random state (so a task like the reproducible-random one doesn't leave NumPy's
    RNG in a different state than it found it in)."""
    namespace = {"np": np}
    namespace.update(_build_dataset_dict())

    rng_state = np.random.get_state()
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    error = None
    old_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                    exec(code_str, namespace)
            finally:
                os.chdir(old_cwd)
    except Exception as e:
        error = f"(Couldn't compute a live preview of the output: {e})"
    finally:
        np.random.set_state(rng_state)

    output = stdout_buf.getvalue().rstrip()
    if len(output) > _MAX_OUTPUT_CHARS:
        output = output[:_MAX_OUTPUT_CHARS] + "\n... (output truncated)"
    return output, error


TASKS = {

    # ============================================================
    # PHASE 1 -- LEARNING
    # ============================================================

    1: {
        "phase": "learning", "title": "Array Creation from Python Lists",
        "hints": ["Both structures can be converted with the same NumPy constructor -- "
                   "pay attention to how a list of lists becomes rows and columns."],
        "answer_code": '''readings_1d = [65.2, 66.8, 64.1, 70.5, 68.9]
arr_1d = np.array(readings_1d)

readings_2d = [[65.2, 66.8], [70.1, 69.4], [63.5, 64.0]]
arr_2d = np.array(readings_2d)

print(arr_1d, type(arr_1d))
print(arr_2d, type(arr_2d))''',
        "explanation": "np.array() converts a flat Python list into a 1D array, and a list "
                        "of equal-length lists into a 2D array, where each inner list becomes a row.",
        "reasoning": "NumPy infers dimensionality from nesting depth, and infers a shared "
                      "dtype from the values -- here, float64 since the values are decimals.",
    },

    2: {
        "phase": "learning", "title": "Creation with zeros / ones / full",
        "hints": ["Three related constructors each take a shape; one of them also takes a fill value."],
        "answer_code": '''alerts = np.zeros(12)
active_flags = np.ones((12, 5))
health_scores = np.full(12, 100)

print(alerts)
print(active_flags)
print(health_scores)''',
        "explanation": "np.zeros and np.ones build arrays of a given shape pre-filled with 0.0 "
                        "or 1.0; np.full takes an explicit shape and fill value.",
        "reasoning": "These are the standard way to pre-allocate arrays before filling them in "
                      "a loop or pipeline step, which is far faster than growing a Python list.",
    },

    3: {
        "phase": "learning", "title": "arange & linspace",
        "hints": ["One function counts by a fixed step size; the other fixes the number of "
                   "output points across a range instead."],
        "answer_code": '''hour_markers = np.arange(0, 120)
thresholds = np.linspace(0, 1, 6)

print(hour_markers)
print(thresholds)''',
        "explanation": "np.arange(start, stop) generates values by step size (default step=1), "
                        "stopping before `stop`. np.linspace(start, stop, n) generates exactly "
                        "n evenly spaced values, including both endpoints by default.",
        "reasoning": "Use arange when you care about the step size; use linspace when you care "
                      "about the exact count of points, e.g. percentile markers.",
    },

    4: {
        "phase": "learning", "title": "Identity Matrix with eye",
        "hints": ["Think of the matrix that leaves a vector unchanged when multiplied by it."],
        "answer_code": '''calibration_baseline = np.eye(5)
print(calibration_baseline)''',
        "explanation": "np.eye(n) builds an n x n identity matrix: ones on the diagonal, "
                        "zeros elsewhere.",
        "reasoning": "Identity matrices are the neutral element for matrix multiplication -- "
                      "a common baseline/default in calibration or transformation pipelines.",
    },

    5: {
        "phase": "learning", "title": "Reproducible Random Arrays",
        "hints": ["One function seeds the generator; the other three each target a different "
                   "distribution or shape."],
        "hint2": "Re-run np.random.seed(7) immediately before each generation call if you want "
                  "that specific call to be reproducible independent of what ran before it.",
        "answer_code": '''np.random.seed(7)
floats = np.random.rand(3, 3)

np.random.seed(7)
ints = np.random.randint(1, 100, 10)

np.random.seed(7)
normal_vals = np.random.randn(4, 4)

print(floats)
print(ints)
print(normal_vals)''',
        "explanation": "np.random.seed(n) fixes the internal state of the random generator. "
                        "rand() draws uniform floats in [0,1), randint(low, high) draws integers "
                        "from [low, high), and randn() draws from the standard normal distribution.",
        "reasoning": "Reproducibility matters for debugging and for tests -- without a seed, "
                      "every run produces different data and bugs become impossible to reliably "
                      "reproduce. In newer code, np.random.default_rng(seed) is the recommended "
                      "generator API, but seed() is still common and fine to know.",
    },

    6: {
        "phase": "learning", "title": "Array Attributes",
        "hints": ["This information is exposed as attributes, not method calls -- no parentheses needed."],
        "answer_code": '''print(temperature.shape)
print(temperature.ndim)
print(temperature.size)
print(temperature.dtype)''',
        "explanation": ".shape returns a tuple of dimension sizes, .ndim the number of "
                        "dimensions, .size the total element count, and .dtype the stored data type.",
        "reasoning": "Checking these attributes is usually the first thing you do with any "
                      "unfamiliar array -- they tell you immediately what you're working with.",
    },

    7: {
        "phase": "learning", "title": "Data Type Casting (astype)",
        "hints": ["Converting dtype produces a new array via a dedicated method; check .nbytes "
                   "before and after to see the memory effect."],
        "answer_code": '''production_f32 = production.astype(np.float32)

pressure_f32 = pressure.astype(np.float32)
print("float64 bytes:", pressure.nbytes)
print("float32 bytes:", pressure_f32.nbytes)''',
        "explanation": ".astype(dtype) returns a new array with the values converted to the "
                        "requested type, leaving the original untouched.",
        "reasoning": "float32 uses half the memory of float64. At production scale (millions of "
                      "sensor readings), this kind of precision/memory trade-off is a real "
                      "engineering decision, not just a syntax exercise.",
    },

    8: {
        "phase": "learning", "title": "1D Indexing",
        "hints": ["Negative indices count backward from the end of the array."],
        "answer_code": '''first_machine = downtime[0]
last_machine = downtime[-1]
print(first_machine, last_machine)''',
        "explanation": "A single integer index retrieves one element; negative indices count "
                        "from the end, so -1 is always the last element regardless of length.",
        "reasoning": "Using -1 instead of len(arr)-1 is more robust to the array's size changing later.",
    },

    9: {
        "phase": "learning", "title": "2D Indexing",
        "hints": ["A single index on a 2D array selects an entire row; two comma-separated "
                   "indices pinpoint one cell."],
        "answer_code": '''machine_3_row = temperature[3]
single_cell = temperature[3, 50]
print(machine_3_row)
print(single_cell)''',
        "explanation": "temperature[3] returns the full 1D row for machine index 3. "
                        "temperature[3, 50] returns the single value at row 3, column 50.",
        "reasoning": "arr[i, j] is preferred over arr[i][j] in NumPy -- it's a single indexing "
                      "operation instead of two, and is what fancy/boolean indexing generalizes from.",
    },

    10: {
        "phase": "learning", "title": "1D Slicing",
        "hints": ["Slicing takes start:stop:step; a negative step reverses direction."],
        "answer_code": '''first_five = downtime[:5]
every_other = downtime[1::2]
reversed_all = downtime[::-1]
print(first_five)
print(every_other)
print(reversed_all)''',
        "explanation": "downtime[:5] takes indices 0-4. downtime[1::2] starts at index 1 and "
                        "takes every second element. downtime[::-1] walks the whole array backward.",
        "reasoning": "Slices return views by default in NumPy (see Task 17), which makes "
                      "slicing very cheap compared to copying.",
    },

    11: {
        "phase": "learning", "title": "2D Slicing",
        "hints": ["Apply a slice to each axis, separated by a comma."],
        "answer_code": '''day1_first4 = temperature[:4, :24]
print(day1_first4.shape)
print(day1_first4)''',
        "explanation": "temperature[:4, :24] slices rows 0-3 (machines) and columns 0-23 "
                        "(day-1 hours) simultaneously.",
        "reasoning": "Each axis gets its own slice; there's no need to slice sequentially.",
    },

    12: {
        "phase": "learning", "title": "Boolean Masking",
        "hints": ["Comparing an array to a scalar produces a same-shaped boolean array you can "
                   "then use to index the original array."],
        "answer_code": '''mask = temperature > 90
hot_readings = temperature[mask]
count = mask.sum()
print(hot_readings)
print("count:", count)''',
        "explanation": "temperature > 90 returns a boolean array of the same shape. Indexing "
                        "with that boolean array pulls out only the True positions as a flat 1D result.",
        "reasoning": "Comparisons involving NaN evaluate to False, so this mask naturally "
                      "excludes any missing readings without extra handling.",
    },

    13: {
        "phase": "learning", "title": "Fancy Indexing",
        "hints": ["Passing a list of positions selects multiple items at once; for 2D "
                   "coordinate pairs, use two parallel lists -- one of row positions, one of column positions."],
        "answer_code": '''selected_machines = downtime[[0, 5, 9]]
print(selected_machines)

cells = temperature[[2, 7, 11], [10, 50, 100]]
print(cells)''',
        "explanation": "downtime[[0,5,9]] pulls three specific elements in one call. "
                        "temperature[[2,7,11],[10,50,100]] pulls the three individual cells "
                        "(2,10), (7,50), and (11,100) -- paired element-wise, not a block.",
        "reasoning": "This paired-list indexing selects coordinates element-wise, which is "
                      "very different from slicing -- temperature[[2,7,11]][:, [10,50,100]] "
                      "would instead give a 3x3 block of all combinations.",
    },

    14: {
        "phase": "learning", "title": "Locating & Replacing with np.where",
        "hints": ["One function returns index locations; the same function can also act as a "
                   "vectorized if/else across a whole array."],
        "answer_code": '''fault_idx = np.where(vibration > 4.5)
print(fault_idx)

vibration_capped = np.where(vibration > 4.5, 4.5, vibration)
print(vibration_capped.max())''',
        "explanation": "np.where(condition) with one argument returns a tuple of index arrays "
                        "where the condition is True. np.where(condition, a, b) returns an array "
                        "that takes values from a where True and from b where False.",
        "reasoning": "The three-argument form is a vectorized equivalent of writing "
                      "`value if condition else other` for every element, without a Python loop.",
    },

    15: {
        "phase": "learning", "title": "Reshaping Arrays",
        "hints": ["reshape lets you specify the new shape directly, and -1 in one position "
                   "can be inferred automatically from the total element count."],
        "answer_code": '''flat = production.reshape(-1)
by_day = flat.reshape(5, 12)
print(flat.shape, by_day.shape)
print(production.size == flat.size == by_day.size)''',
        "explanation": "reshape(-1) flattens to 1D. reshape(5, 12) then reorganizes the same 60 "
                        "values into 5 rows of 12. -1 tells NumPy to infer that dimension so it "
                        "doesn't have to be computed by hand.",
        "reasoning": "reshape never changes the total element count or the underlying data "
                      "values -- only how they're grouped into axes.",
    },

    16: {
        "phase": "learning", "title": "Flatten vs Ravel",
        "hints": ["One approach guarantees a brand-new array; the other tries to avoid "
                   "copying whenever it can. Work on a copy of the dataset so you don't "
                   "disturb the shared arrays used by later tasks."],
        "answer_code": '''temp_sandbox = temperature.copy()   # keep the shared dataset untouched by this demo

flat_copy = temp_sandbox.flatten()
flat_copy[0] = -999
print("flatten() source affected?", temp_sandbox.flat[0] == -999)   # False

flat_view = temp_sandbox.ravel()
flat_view[0] = -999
print("ravel() source affected?", temp_sandbox.flat[0] == -999)     # True
''',
        "explanation": ".flatten() always allocates new memory, so modifying the result never "
                        "affects the source. .ravel() returns a view sharing memory with the "
                        "source whenever the data layout allows it, so modifying the result can "
                        "silently modify the source too.",
        "reasoning": "Prefer ravel() when you only need to read the flattened data (it's "
                      "cheaper); prefer flatten() when you intend to modify the result "
                      "independently. Working on temp_sandbox here (instead of temperature "
                      "directly) keeps this demonstration from corrupting the shared dataset "
                      "used throughout the rest of the workbook.",
    },

    17: {
        "phase": "learning", "title": "View vs Copy",
        "hints": ["Basic slicing in NumPy does not duplicate the underlying data by default. "
                   "Work on a copy of the dataset so you don't disturb the shared arrays used "
                   "by later tasks."],
        "hint2": "Compare demo[:3] (a view) against demo[:3].copy() (a real copy) by mutating "
                  "each and checking the source array afterward.",
        "answer_code": '''pressure_demo = pressure.copy()   # keep the shared dataset untouched by this demo

subset = pressure_demo[:3]
subset[0, 0] = 999
print("plain slice -- source affected?", pressure_demo[0, 0] == 999)   # True

pressure_demo2 = pressure.copy()
subset2 = pressure_demo2[:3].copy()
subset2[0, 0] = 111
print("explicit .copy() -- source affected?", pressure_demo2[0, 0] == 111)   # False''',
        "explanation": "Basic slicing (start:stop:step on plain axes) returns a view that "
                        "shares memory with the source array. .copy() explicitly allocates new "
                        "memory, so changes to the copy never touch the original.",
        "reasoning": "This is one of the most common real-world NumPy bugs -- unintentionally "
                      "mutating shared data through a view. Note that fancy indexing (lists) and "
                      "boolean indexing both return copies, unlike basic slicing -- the rule "
                      "depends on which indexing style you use. Working on pressure_demo here "
                      "(instead of pressure directly) keeps this demonstration from corrupting "
                      "the shared dataset used throughout the rest of the workbook.",
    },

    18: {
        "phase": "learning", "title": "Elementwise Arithmetic",
        "hints": ["Arithmetic operators apply to same-shaped arrays position by position, "
                   "with no explicit loop needed."],
        "answer_code": '''stress_index = vibration * 10 + (pressure - 100)
print(stress_index.shape)
print(stress_index[:2, :5])''',
        "explanation": "+, -, *, / between two same-shaped arrays apply elementwise, producing "
                        "a result of the same shape.",
        "reasoning": "This is the foundation of vectorized computation -- combining several "
                      "arithmetic operations in one expression still runs as a handful of fast, "
                      "compiled passes over the data rather than a Python loop.",
    },

    19: {
        "phase": "learning", "title": "Broadcasting (scalar & 1D-2D)",
        "hints": ["A scalar operation stretches across every element; a 1D array whose length "
                   "matches the trailing axis of a 2D array stretches across that axis."],
        "answer_code": '''temperature_f = temperature * 9 / 5 + 32

daily_targets = np.array([200, 205, 210, 195, 220])
production_adj = production + daily_targets
print(production_adj.shape)''',
        "explanation": "The scalar formula applies to every one of the 12x120 elements at "
                        "once. daily_targets has shape (5,), matching production's trailing "
                        "axis (also 5), so it's broadcast across every row.",
        "reasoning": "Broadcasting compares shapes starting from the last (rightmost) "
                      "dimension -- (12,5) and (5,) are compatible because the trailing 5s match.",
    },

    20: {
        "phase": "learning", "title": "Broadcasting Across Rows (Calibration)",
        "hints": ["To broadcast along rows instead of the trailing axis, the 1D array "
                   "typically needs to be reshaped into a column first."],
        "hint2": "A shape of (12,) added to (12,120) won't align on the trailing axis (120 vs "
                  "12). Reshape the offsets to (12,1) so broadcasting stretches it across "
                  "each row's 120 columns instead.",
        "answer_code": '''calibration_offsets = np.random.uniform(-1, 1, 12).round(2)
temperature_calibrated = temperature + calibration_offsets.reshape(-1, 1)
print(temperature_calibrated.shape)''',
        "explanation": "calibration_offsets.reshape(-1, 1) turns a (12,) array into a (12,1) "
                        "column. Adding that to a (12,120) array broadcasts the single value in "
                        "each row across all 120 columns of that row.",
        "reasoning": "Whether broadcasting happens 'down columns' or 'across rows' depends "
                      "entirely on which axis has size 1 -- this is a very common point of "
                      "confusion worth being deliberate about.",
    },

    21: {
        "phase": "learning", "title": "Aggregate Statistics",
        "hints": ["Standard aggregate functions can be called directly on the whole array, "
                   "collapsing it to one number."],
        "answer_code": '''print("mean:", pressure.mean())
print("min:", pressure.min())
print("max:", pressure.max())
print("std:", pressure.std())''',
        "explanation": ".mean(), .min(), .max(), and .std() with no axis argument collapse "
                        "the entire array (all machines, all hours) into a single scalar.",
        "reasoning": "These are called on the whole array here because pressure has no NaN "
                      "values except the injected zero-faults, which are valid numbers -- "
                      "min() correctly reports 0 as the lowest recorded (faulty) reading.",
    },

    22: {
        "phase": "learning", "title": "Axis-based Aggregation",
        "hints": ["axis=1 collapses across columns (days), axis=0 collapses across rows "
                   "(machines) -- pick the one that matches what you're averaging over."],
        "answer_code": '''avg_per_machine = production.mean(axis=1)
avg_per_day = production.mean(axis=0)
hottest_producer = np.argmax(avg_per_machine)

print(avg_per_machine)
print(avg_per_day)
print("highest-output machine index:", hottest_producer)''',
        "explanation": "axis=1 averages across the 5 days for each machine, leaving one value "
                        "per machine (12 values). axis=0 averages across the 12 machines for "
                        "each day, leaving one value per day (5 values).",
        "reasoning": "A useful mental shortcut: the axis you pass is the one that "
                      "disappears/collapses in the output shape.",
    },

    23: {
        "phase": "learning", "title": "Cumulative Functions",
        "hints": ["One function accumulates values progressively along an axis; combine it "
                   "with a boolean comparison to find a crossing point."],
        "answer_code": '''cum_production = np.cumsum(production, axis=1)
crossed = cum_production >= 600
first_day_crossed = np.argmax(crossed, axis=1)

print(cum_production)
print(first_day_crossed)''',
        "explanation": "np.cumsum(axis=1) produces a running total across each machine's 5 "
                        "days. Comparing to 600 gives a boolean array; argmax on a boolean array "
                        "returns the index of the first True (since True sorts higher than False).",
        "reasoning": "This argmax-on-boolean pattern is a common idiom for 'find the first "
                      "position where a condition becomes true' without an explicit loop.",
    },

    24: {
        "phase": "learning", "title": "Clipping Values (np.clip)",
        "hints": ["There's a dedicated function for bounding every value within a min and max, "
                   "without writing conditional logic yourself."],
        "answer_code": '''vibration_clean = np.clip(vibration, 0, 5.0)
print(vibration_clean.max(), vibration_clean.min())''',
        "explanation": "np.clip(array, low, high) caps every value to stay within [low, high], "
                        "leaving in-range values untouched.",
        "reasoning": "This is a fast, vectorized alternative to writing "
                      "np.where(x > high, high, np.where(x < low, low, x)) by hand.",
    },

    25: {
        "phase": "learning", "title": "Unique Values & Counts",
        "hints": ["One function both finds distinct values and, with an option, counts how "
                   "often each occurs."],
        "answer_code": '''rounded = np.round(pressure).astype(int)
values, counts = np.unique(rounded, return_counts=True)
print(values)
print(counts)''',
        "explanation": "np.unique returns the sorted distinct values in the array. Passing "
                        "return_counts=True additionally returns how many times each value occurs.",
        "reasoning": "This is a quick way to build a frequency table directly from an array "
                      "without needing pandas.",
    },

    26: {
        "phase": "learning", "title": "any / all for Validation Checks",
        "hints": ["Two functions test whether at least one, or every single, element of a "
                   "boolean array satisfies a condition."],
        "answer_code": '''has_zero_downtime = np.any(downtime == 0)
all_hit_150 = np.all(production >= 150)

print(has_zero_downtime)
print(all_hit_150)''',
        "explanation": "np.any(condition) returns True if at least one element satisfies the "
                        "condition. np.all(condition) returns True only if every element does.",
        "reasoning": "These are commonly used as fast validation/assertion checks on data "
                      "before running an expensive downstream computation.",
    },

    27: {
        "phase": "learning", "title": "Handling Missing Data",
        "hints": ["Standard aggregate functions propagate NaN into the result; NumPy has "
                   "NaN-aware counterparts, plus a direct way to detect NaN positions."],
        "hint2": "np.isnan(array) gives a boolean mask of missing positions; np.nanmean "
                  "ignores NaN when averaging; np.where can then substitute values only at "
                  "those masked positions.",
        "answer_code": '''nan_count = np.isnan(temperature).sum()
clean_mean = np.nanmean(temperature)
temperature_filled = np.where(np.isnan(temperature), clean_mean, temperature)

print("missing readings:", nan_count)
print("mean ignoring NaN:", clean_mean)
print("any NaN remaining:", np.isnan(temperature_filled).any())''',
        "explanation": "np.isnan() flags missing positions. temperature.mean() would return "
                        "nan because any single NaN poisons a normal aggregate; np.nanmean() "
                        "skips NaN values instead. np.where fills only the NaN positions with "
                        "the clean mean, leaving valid readings untouched.",
        "reasoning": "Contrast this with temperature.mean(), which silently returns nan -- a "
                      "very common production bug where a single bad sensor reading corrupts an "
                      "entire downstream report without an obvious error.",
    },

    28: {
        "phase": "learning", "title": "Sorting",
        "hints": ["Default sort order is ascending; reversing the sorted result gives "
                   "descending order."],
        "answer_code": '''ascending = np.sort(downtime)
descending = np.sort(downtime)[::-1]
print(ascending)
print(descending)''',
        "explanation": "np.sort() returns a new sorted array (ascending by default) without "
                        "modifying the original. Reversing with [::-1] gives descending order.",
        "reasoning": "np.sort() returns a copy; the array's own .sort() method sorts in place "
                      "instead -- worth knowing which one you're calling.",
    },

    29: {
        "phase": "learning", "title": "Argsort & Ranking",
        "hints": ["One function returns the positions that would sort the array, rather than "
                   "the sorted values themselves."],
        "answer_code": '''order = np.argsort(downtime)[::-1]
top3_worst = order[:3]
print(top3_worst)
print(downtime[top3_worst])''',
        "explanation": "np.argsort() returns the indices that would put the array in ascending "
                        "order. Reversing that index array before slicing the first 3 gives the "
                        "machines with the highest downtime, worst first.",
        "reasoning": "argsort is essential whenever you need to rank items but still care "
                      "about which original entity each rank belongs to.",
    },

    30: {
        "phase": "learning", "title": "Axis-based Sorting (2D)",
        "hints": ["Axis-based sorting works the same way axis-based aggregation does -- choose "
                   "the direction you want to sort along."],
        "answer_code": '''sorted_within_machine = np.sort(production, axis=1)
sorted_within_day = np.sort(production, axis=0)
print(sorted_within_machine)
print(sorted_within_day)''',
        "explanation": "axis=1 sorts each row independently (each machine's own 5 days get "
                        "reordered). axis=0 sorts each column independently (each day's 12 "
                        "machine values get reordered).",
        "reasoning": "Sorting along an axis never moves values between rows or columns -- it "
                      "only reorders values within that axis.",
    },

    31: {
        "phase": "learning", "title": "Concatenation",
        "hints": ["Choose the axis that lines up with 'columns represent days.'"],
        "answer_code": '''new_days = np.random.randint(180, 260, size=(12, 2))
production_extended = np.concatenate([production, new_days], axis=1)
print(production_extended.shape)''',
        "explanation": "np.concatenate joins arrays along an existing axis. Since days are "
                        "columns here, axis=1 appends the new 2 columns after the existing 5, "
                        "producing shape (12, 7).",
        "reasoning": "All dimensions except the join axis must match exactly, or concatenate "
                      "raises an error -- both arrays needed 12 rows here.",
    },

    32: {
        "phase": "learning", "title": "Vertical / Horizontal Stacking",
        "hints": ["One stacking function adds along rows, the other along columns; a plain "
                   "1D row may need reshaping first."],
        "answer_code": '''machine13 = np.random.randint(180, 260, size=5)
production_with_new_machine = np.vstack([production, machine13])
print(production_with_new_machine.shape)   # (13, 5)

batch2 = np.random.randint(180, 260, size=(12, 5))
side_by_side = np.hstack([production, batch2])
print(side_by_side.shape)   # (12, 10)''',
        "explanation": "np.vstack stacks arrays on top of each other (adds rows); "
                        "np.hstack places arrays side by side (adds columns). vstack "
                        "automatically treats a 1D array as one new row.",
        "reasoning": "vstack/hstack are convenience wrappers around concatenate for the most "
                      "common row/column-adding cases.",
    },

    33: {
        "phase": "learning", "title": "Splitting Arrays",
        "hints": ["There's a function that divides an array into N equal parts along a "
                   "chosen axis."],
        "answer_code": '''zones = np.split(production, 3, axis=0)
for i, zone in enumerate(zones):
    print(f"Zone {i}:\\n", zone)''',
        "explanation": "np.split(array, n, axis) divides the array into n equal pieces along "
                        "the given axis. Here, 12 machines split 3 ways gives 3 groups of 4.",
        "reasoning": "np.split requires the axis length to divide evenly, or it raises an "
                      "error; np.array_split tolerates uneven division and is the safer choice "
                      "when group sizes aren't guaranteed to divide evenly.",
    },

    34: {
        "phase": "learning", "title": "Tile & Repeat",
        "hints": ["One function repeats the entire array as a whole block; the other repeats "
                   "each individual element in place."],
        "answer_code": '''shift_pattern = np.array([1, 2, 3])
tiled = np.tile(shift_pattern, 4)
repeated = np.repeat(shift_pattern, 4)

print(tiled)      # [1 2 3 1 2 3 1 2 3 1 2 3]
print(repeated)   # [1 1 1 1 2 2 2 2 3 3 3 3]''',
        "explanation": "np.tile repeats the whole array as a block n times. np.repeat repeats "
                        "each element n times in place, keeping elements together instead of "
                        "the pattern.",
        "reasoning": "These produce very different orderings from the same input -- choose "
                      "based on whether you want the pattern repeated or each value expanded.",
    },

    35: {
        "phase": "learning", "title": "Linear Algebra Basics",
        "hints": ["Think of this as combining each row of production with energy_weights "
                   "into a single weighted sum -- a standard linear algebra operation."],
        "answer_code": '''energy_weights = np.array([1.2, 1.0, 1.1, 1.3, 0.9])
total_energy = production @ energy_weights
print(total_energy.shape)   # (12,)
print(total_energy)''',
        "explanation": "@ (equivalently np.matmul or np.dot for these shapes) performs matrix "
                        "multiplication. A (12,5) array combined with a (5,) array contracts "
                        "over the shared dimension of 5, producing 12 results.",
        "reasoning": "This is equivalent to computing a per-row weighted dot product manually "
                      "in a loop, but as a single vectorized operation.",
    },

    36: {
        "phase": "learning", "title": "Multi-dimensional Arrays (3D Stacking)",
        "hints": ["One stacking function adds a brand-new leading axis; axis-based "
                   "aggregation then works the same way, just with an extra axis to choose from."],
        "answer_code": '''sensor_cube = np.stack([temperature, vibration, pressure], axis=0)
print(sensor_cube.shape)   # (3, 12, 120)

mean_per_sensor = np.nanmean(sensor_cube, axis=(1, 2))
mean_per_machine = np.nanmean(sensor_cube, axis=(0, 2))
print(mean_per_sensor)
print(mean_per_machine)''',
        "explanation": "np.stack([a, b, c], axis=0) creates a new leading axis of length 3, "
                        "one slot per input array, producing shape (3, 12, 120). axis can be a "
                        "tuple, so axis=(1,2) collapses machines and hours together, leaving one "
                        "value per sensor type.",
        "reasoning": "np.stack differs from np.concatenate: stack always adds a new axis, "
                      "while concatenate joins along an axis that already exists.",
    },

    37: {
        "phase": "learning", "title": "Vectorization vs Python Loops",
        "hints": ["Any operation you can express as array arithmetic is almost always faster "
                   "than iterating in Python -- measure it rather than assuming."],
        "hint2": "Use time.perf_counter() before and after each version, or %timeit in a "
                  "notebook cell, and compare the two durations directly.",
        "answer_code": '''import time

start = time.perf_counter()
loop_result = []
for row in temperature:
    row_result = []
    for val in row:
        row_result.append((val - 65) ** 2)
    loop_result.append(row_result)
loop_time = time.perf_counter() - start

start = time.perf_counter()
vectorized_result = (temperature - 65) ** 2
vector_time = time.perf_counter() - start

print("loop time:", loop_time)
print("vectorized time:", vector_time)''',
        "explanation": "Both versions compute the same squared-deviation values. The loop "
                        "version processes one Python float at a time; the vectorized version "
                        "runs the whole operation as compiled array arithmetic.",
        "reasoning": "The vectorized version is dramatically faster because it avoids Python's "
                      "per-element interpreter overhead -- the gap grows as the array size grows, "
                      "which is exactly the situation production data pipelines face.",
    },

    38: {
        "phase": "learning", "title": "Saving & Loading Arrays",
        "hints": ["One function saves/loads a single array; a related function bundles "
                   "multiple named arrays into one file."],
        "answer_code": '''np.save("production.npy", production)
loaded = np.load("production.npy")
print(np.array_equal(production, loaded))

np.savez_compressed("sensors.npz", temperature=temperature, vibration=vibration, pressure=pressure)
data = np.load("sensors.npz")
print(data["temperature"].shape)''',
        "explanation": "np.save/np.load handle a single array in NumPy's native .npy binary "
                        "format. np.savez / np.savez_compressed bundle several named arrays "
                        "into one .npz archive, retrievable by name after loading.",
        "reasoning": ".npy/.npz preserve dtype and shape exactly and load much faster than "
                      "text formats like CSV -- important when moving array data between stages "
                      "of a data pipeline.",
    },

    # ============================================================
    # PHASE 2 -- TESTING
    # ============================================================

    39: {
        "phase": "testing", "title": "Rebuild the Reading",
        "hints": ["Think about which NumPy attribute reports element count directly, without "
                   "calling a Python built-in."],
        "answer_code": '''readings = [2.1, 2.4, 2.0, 2.6, 2.3, 2.5, 2.2, 2.1, 2.7, 2.4, 2.3, 2.2,
            2.5, 2.6, 2.1, 2.0, 2.4, 2.3, 2.2, 2.5, 2.6, 2.4, 2.3, 2.1]
arr = np.array(readings)
print(arr.size)''',
        "explanation": "np.array() converts the list, and .size reports the element count "
                        "directly as an attribute.",
        "reasoning": "len() would also work for a 1D array, but .size generalizes correctly to "
                      "any number of dimensions, which len() does not (len() only gives the "
                      "size of the first axis).",
    },

    40: {
        "phase": "testing", "title": "Describe the Array",
        "hints": ["Three separate attributes, each answering one specific question about the array."],
        "answer_code": '''n_machines_covered = pressure.shape[0]
readings_per_machine = pressure.shape[1]
precision = pressure.dtype
print(n_machines_covered, readings_per_machine, precision)''',
        "explanation": "shape[0] is the machine count, shape[1] is the readings-per-machine "
                        "count, and dtype reports the numeric precision.",
        "reasoning": "Reading .shape as (rows, columns) = (machines, hours) here relies on "
                      "knowing how the array was constructed -- shape alone doesn't label axes, "
                      "which is why documenting array layout matters in real pipelines.",
    },

    41: {
        "phase": "testing", "title": "Grab the Weekend",
        "hints": ["Slice along the hour axis using the given boundary as the start point."],
        "answer_code": '''final_day = temperature[:, 96:120]
print(final_day.shape)   # (12, 24)''',
        "explanation": "Slicing columns 96 through 119 (120 is exclusive) isolates the final "
                        "24-hour block for every machine.",
        "reasoning": "The row slice `:` means 'keep all machines,' while the column slice "
                      "restricts to the target hour range.",
    },

    42: {
        "phase": "testing", "title": "Flag the Risk",
        "hints": ["Combine two conditions with a bitwise operator before using the result to index."],
        "answer_code": '''mask = (pressure < 95) | (pressure > 108)
offending_values = pressure[mask]
print(offending_values)''',
        "explanation": "Each condition produces its own boolean array; | combines them "
                        "elementwise as OR. The combined mask then selects any reading outside "
                        "the valid range.",
        "reasoning": "Python's `or`/`and` don't work elementwise on arrays -- NumPy requires "
                      "the bitwise operators | and & instead, and each condition needs its own "
                      "parentheses because of operator precedence.",
    },

    43: {
        "phase": "testing", "title": "One Number to Rule Them All",
        "hints": ["No axis argument is needed when you want a single overall answer."],
        "answer_code": '''overall_max = vibration.max()
print(overall_max)''',
        "explanation": "Calling .max() with no axis collapses the entire 2D array to one scalar.",
        "reasoning": "Omitting axis is the signal that you want a single number across every "
                      "dimension at once.",
    },

    44: {
        "phase": "testing", "title": "Same Data, New Shape",
        "hints": ["You don't have to calculate every new dimension yourself."],
        "answer_code": '''restructured = production.reshape(3, -1)
print(restructured.shape)   # (3, 20)''',
        "explanation": "reshape(3, -1) fixes the row count at 3 and lets NumPy compute the "
                        "column count needed to preserve all 60 elements.",
        "reasoning": "-1 is useful whenever you know one dimension and want the other derived "
                      "automatically, avoiding a manual division that could be wrong.",
    },

    45: {
        "phase": "testing", "title": "Line Them Up",
        "hints": ["The default ordering direction is ascending."],
        "answer_code": '''ordered_downtime = np.sort(downtime)
print(ordered_downtime)''',
        "explanation": "np.sort() with no arguments returns values from lowest to highest.",
        "reasoning": "The question asks for the values themselves, not which machine they "
                      "belong to, so plain sort (not argsort) is the right tool here.",
    },

    46: {
        "phase": "testing", "title": "Worst Hour of the Week",
        "hints": ["Think about which function locates the position of an extreme value, and "
                   "whether the ordinary version handles missing data safely."],
        "answer_code": '''worst_hour_per_machine = np.nanargmax(temperature, axis=1)
print(worst_hour_per_machine)''',
        "explanation": "np.nanargmax behaves like argmax but ignores NaN values when locating "
                        "the maximum, returning valid results even for rows containing missing readings.",
        "reasoning": "Plain np.argmax can return unreliable results in the presence of NaN "
                      "(NaN participates unpredictably in comparisons), so the NaN-aware variant "
                      "is the safe default whenever missing data is possible.",
    },

    47: {
        "phase": "testing", "title": "Fair Comparison",
        "hints": ["You need a per-row baseline subtracted from every value in that row -- "
                   "think about what shape the baseline needs before the subtraction aligns correctly."],
        "answer_code": '''baseline = np.nanmean(temperature, axis=1, keepdims=True)
deviation = temperature - baseline
print(deviation.shape)   # (12, 120), same as temperature''',
        "explanation": "axis=1, keepdims=True computes each machine's own average but keeps "
                        "the result as shape (12,1) instead of collapsing to (12,). That shape "
                        "broadcasts cleanly against the original (12,120) array.",
        "reasoning": "Without keepdims, the mean would come back as shape (12,), which "
                      "broadcasts against the wrong axis (columns instead of rows) and produces "
                      "an incorrect or error-raising result.",
    },

    48: {
        "phase": "testing", "title": "Trustworthy Average",
        "hints": ["Think about what a single missing value does to a standard aggregate function."],
        "answer_code": '''broken = temperature.mean()          # likely nan
fixed = np.nanmean(temperature)      # correct
print(broken, fixed)''',
        "explanation": "temperature.mean() returns nan because temperature contains missing "
                        "(NaN) values, and any arithmetic involving NaN propagates NaN. "
                        "np.nanmean() skips NaN entries and computes the mean over valid values only.",
        "reasoning": "This is why NaN-aware functions exist as a separate family -- ordinary "
                      "aggregates silently break the moment any NaN is present anywhere in the array.",
    },

    49: {
        "phase": "testing", "title": "Zone Performance",
        "hints": ["Slice each machine group before aggregating; think about which axis to "
                   "collapse first, and what a sanity check on your slices would look like."],
        "answer_code": '''zone_a = production[0:4].mean()
zone_b = production[4:8].mean()
zone_c = production[8:12].mean()
print(zone_a, zone_b, zone_c)

# sanity check: confirm the three slices cover every machine exactly once
assert 4 + 4 + 4 == production.shape[0]''',
        "explanation": "Each zone is isolated with a row slice before calling .mean(), which "
                        "collapses both machines and days within that zone to one number.",
        "reasoning": "A meaningful sanity check here is confirming the slices are "
                      "non-overlapping and collectively cover all 12 machines -- an off-by-one "
                      "slicing error would silently produce a plausible-looking wrong answer.",
    },

    50: {
        "phase": "testing", "title": "Cherry-Pick the Flags",
        "hints": ["Separate the coordinate pairs into two parallel lists before indexing."],
        "answer_code": '''coords = [(1, 15), (4, 88), (9, 40), (11, 5)]
rows = [c[0] for c in coords]
cols = [c[1] for c in coords]
flagged_readings = vibration[rows, cols]
print(flagged_readings)''',
        "explanation": "Splitting the coordinate pairs into a row-index list and a "
                        "column-index list lets a single fancy-indexing call retrieve all four "
                        "specific cells at once.",
        "reasoning": "This mirrors Task 13's pattern -- paired-list indexing pulls exact "
                      "coordinates, not a rectangular block.",
    },

    51: {
        "phase": "testing", "title": "Sensor Sanity Check",
        "hints": ["You'll need a condition, a per-row replacement value, and a function that "
                   "chooses between two arrays element by element."],
        "answer_code": '''row_valid_means = np.array([
    pressure[i][pressure[i] != 0].mean() for i in range(pressure.shape[0])
])
pressure_fixed = np.where(pressure == 0, row_valid_means.reshape(-1, 1), pressure)
print((pressure_fixed == 0).any())   # False -- no zero faults remain''',
        "explanation": "Each machine's own valid (non-zero) mean is computed first, then "
                        "np.where replaces only the 0-valued faults with that row's mean, "
                        "leaving genuine readings untouched.",
        "reasoning": "A small Python loop over just 12 machines is a reasonable choice here, "
                      "since each row can have a different number of valid values -- a fully "
                      "vectorized version would need masked arrays and would likely be harder to "
                      "read for very little benefit at this scale (see Task 64).",
    },

    52: {
        "phase": "testing", "title": "Merge the Batches",
        "hints": ["For the first case, think about which axis represents 'machines.' The "
                   "second case follows the same pattern already used for extending days."],
        "answer_code": '''batch_a = production[0:6]
batch_b = production[6:12]
recombined_machines = np.concatenate([batch_a, batch_b], axis=0)
print(recombined_machines.shape)   # (12, 5)

days_part1 = production[:, :3]
days_part2 = production[:, 3:]
recombined_days = np.concatenate([days_part1, days_part2], axis=1)
print(recombined_days.shape)   # (12, 5)''',
        "explanation": "Joining along axis=0 stacks additional machines (rows); joining along "
                        "axis=1 stacks additional days (columns) -- the axis choice depends on "
                        "which dimension is being extended.",
        "reasoning": "np.vstack/np.hstack would also work here and are equivalent to "
                      "concatenate with axis=0/axis=1 respectively for 2D arrays.",
    },

    53: {
        "phase": "testing", "title": "Find the Bug -- Broadcasting",
        "hints": ["Check that the length of the 1D array actually lines up with the axis you "
                   "intend it to broadcast against."],
        "hint2": "production has shape (12, 5) -- 5 days. daily_bonus needs exactly 5 values "
                  "to broadcast against the trailing axis, not 6.",
        "answer_code": '''# Bug: daily_bonus has 6 elements but production has only 5 columns (5 days)
daily_bonus = np.array([5, 5, 10, 10, 15])   # fixed: 5 elements
production_with_bonus = production + daily_bonus
print(production_with_bonus.shape)''',
        "explanation": "The original daily_bonus array had 6 elements while production's "
                        "trailing axis (days) has 5 -- broadcasting requires the trailing "
                        "dimensions to match (or one of them to be 1), so NumPy raised a ValueError.",
        "reasoning": "Broadcasting failures are almost always a shape mismatch on the trailing "
                      "axis -- always check .shape on both operands first when debugging one.",
    },

    54: {
        "phase": "testing", "title": "Find the Bug -- Wrong Axis",
        "hints": ["Think about what axis=1 collapses versus axis=0, given the array's shape "
                   "is (machines, hours)."],
        "answer_code": '''# Original code computed the average PER MACHINE (12 values), not per hour
per_machine_avg = np.nanmean(temperature, axis=1)   # what the original code actually did

# Corrected: average across machines, for each hour
hourly_avg = np.nanmean(temperature, axis=0)
print(hourly_avg.shape)   # (120,) -- one value per hour, as intended''',
        "explanation": "axis=1 collapses the hours axis, leaving one value per machine (12 "
                        "results) -- the opposite of what was intended. axis=0 collapses the "
                        "machines axis instead, correctly leaving one value per hour (120 results).",
        "reasoning": "The axis argument names the dimension that disappears from the output, "
                      "not the dimension you want to keep -- a frequent source of off-by-one-axis bugs.",
    },

    55: {
        "phase": "testing", "title": "Build the Anomaly Report",
        "hints": ["Break the problem into pieces: a boolean array, a per-row check, a per-row "
                   "count, and a way to find which row has the maximum count."],
        "answer_code": '''anomaly_mask = vibration > 4.5
per_machine_has_anomaly = anomaly_mask.any(axis=1)
total_anomalies = anomaly_mask.sum()
per_machine_count = anomaly_mask.sum(axis=1)
worst_machine = np.argmax(per_machine_count)

print(per_machine_has_anomaly)
print(total_anomalies)
print(per_machine_count)
print(worst_machine)''',
        "explanation": "A single boolean mask over the whole array is reused for every part of "
                        "the report: .any(axis=1) for the yes/no flag per machine, .sum() for "
                        "the fleet-wide total, .sum(axis=1) for the per-machine count, and "
                        "argmax on that count for the worst offender.",
        "reasoning": "Building one boolean mask and reusing it for several aggregations is "
                      "more efficient and more readable than recomputing the condition repeatedly.",
    },

    56: {
        "phase": "testing", "title": "Find the Bug -- Silent Mutation",
        "hints": ["Consider what basic indexing like data[0] actually returns, and whether an "
                   "in-place assignment on it can reach back into the source array."],
        "hint2": "data[0] is a view sharing memory with data. Assigning into that view with "
                  "row[mask] = 0 writes directly into the original array's memory.",
        "answer_code": '''def normalize_first_row(data):
    row = data[0].copy()          # fix: explicit copy breaks the shared-memory link
    row[np.isnan(row)] = 0
    return (row - row.mean()) / row.std()

result = normalize_first_row(temperature)
print("Normalized row sample:", result[:5])
print("Original temperature[0, 0] unaffected:", temperature[0, 0])''',
        "explanation": "data[0] is basic indexing, which returns a view rather than a copy. "
                        "The in-place assignment row[np.isnan(row)] = 0 therefore mutates the "
                        "original array's memory directly, corrupting it for every future call.",
        "reasoning": "Adding .copy() breaks the shared-memory link so the function's internal "
                      "cleanup can never leak out and affect the caller's data -- a defensive "
                      "habit worth having at function boundaries (see Task 63).",
    },

    57: {
        "phase": "testing", "title": "Speed It Up",
        "hints": ["Think about how a boolean comparison across the whole 2D array, combined "
                   "with axis-based aggregation, could replace both nested loops at once."],
        "answer_code": '''def count_exceedances_fast(data, threshold):
    return (data > threshold).sum(axis=1)

result = count_exceedances_fast(vibration, 3.0)
print(result)''',
        "explanation": "data > threshold produces a boolean array of the same shape in one "
                        "pass; .sum(axis=1) then counts True values per row -- replacing both "
                        "nested Python loops with two vectorized operations.",
        "reasoning": "The original version pays Python-level interpreter overhead for every "
                      "single element visited; the vectorized version pushes both the comparison "
                      "and the summation into compiled C loops, which is dramatically faster at scale.",
    },

    58: {
        "phase": "testing", "title": "Pick the Right Tool",
        "hints": ["Think about what each approach actually gives you back -- values versus "
                   "positions -- and which one the question is actually asking for."],
        "answer_code": '''best_machine_per_hour = np.argmax(pressure, axis=0)
print(best_machine_per_hour.shape)   # (120,)''',
        "explanation": "argmax(axis=0) directly returns the row (machine) index of the "
                        "maximum value in each column (hour), which is exactly what's being asked.",
        "reasoning": "The sorting-based alternative (np.argsort(pressure, axis=0)[-1]) also "
                      "works but does unnecessary extra work -- a full O(n log n) sort just to "
                      "extract a single extreme value, where argmax does the same job in O(n).",
    },

    59: {
        "phase": "testing", "title": "Empty Handed",
        "hints": ["Think about what shape and length a boolean-filtered result has when the "
                   "condition matches nothing at all."],
        "answer_code": '''impossible = temperature[temperature > 150]
print(impossible, impossible.shape)   # array([]), shape (0,)

if impossible.size > 0:
    print(impossible.mean())
else:
    print("No readings exceeded the threshold -- nothing to average.")''',
        "explanation": "When no elements satisfy the condition, boolean indexing returns a "
                        "valid but empty array of shape (0,), not an error and not None.",
        "reasoning": "Calling .mean() directly on an empty array returns nan along with a "
                      "RuntimeWarning rather than raising an exception -- checking .size before "
                      "aggregating avoids silently reporting a meaningless nan as if it were a real result.",
    },

    60: {
        "phase": "testing", "title": "What Does This Ranking Mean",
        "hints": ["Distinguish between what argsort's output values represent versus their "
                   "positions in the result."],
        "answer_code": '''order = np.argsort(downtime)
print(order)
# order[0] is the index of the SMALLEST downtime value, not the largest.
print("machine with lowest downtime:", order[0])
print("machine with highest downtime:", order[-1])''',
        "explanation": "The teammate's interpretation is incorrect. argsort returns indices in "
                        "ascending order of the underlying values by default, so the first index "
                        "in the result (3, in their example) belongs to the machine with the "
                        "LOWEST downtime, not the highest.",
        "reasoning": "To find the highest, either reverse the argsort result and take the "
                      "first entry, or look at the last entry of the ascending result directly.",
    },

    61: {
        "phase": "testing", "title": "Slice Through the Cube",
        "hints": ["Three axes means three index positions; a single index on one axis "
                   "collapses that dimension out of the result."],
        "answer_code": '''sensor_cube = np.stack([temperature, vibration, pressure], axis=0)
machine5_vibration = sensor_cube[1, 5, :]
print(machine5_vibration.shape)   # (120,)''',
        "explanation": "Index 1 on the first axis selects the vibration slice (sensor order "
                        "was [temperature, vibration, pressure]); index 5 on the second axis "
                        "selects machine 5; the trailing : keeps all 120 hours.",
        "reasoning": "Indexing a single position on an axis (rather than slicing it) removes "
                      "that axis from the result's shape entirely, dropping the result from 3D to 1D.",
    },

    62: {
        "phase": "testing", "title": "Why Did Broadcasting Fail?",
        "hints": ["Think about how shapes are compared starting from the rightmost dimension, "
                   "and what condition must hold true at each position."],
        "answer_code": '''# Example of shapes that CANNOT broadcast:
a = np.zeros((4, 3))
b = np.zeros((4,))
try:
    a + b
except ValueError as e:
    print("a + b fails:", e)

# Example of shapes that CAN broadcast:
c = np.zeros((3,))
result = a + c
print("a + c succeeds, result shape:", result.shape)''',
        "explanation": "NumPy compares two shapes starting from the rightmost dimension and "
                        "working left. At each position, the dimensions must either be equal or "
                        "one of them must be 1 (missing leading dimensions are treated as 1). If "
                        "any position fails both conditions, broadcasting fails.",
        "reasoning": "(4,3) and (4,) fail because the trailing dimensions (3 vs 4) are unequal "
                      "and neither is 1. (4,3) and (3,) succeed because the trailing dimensions "
                      "(3 vs 3) match exactly.",
    },

    63: {
        "phase": "testing", "title": "View, Copy, and the Cost of Being Wrong",
        "hints": ["Think about both the memory/performance cost and the correctness risk of "
                   "each choice, and when each one matters more."],
        "answer_code": None,
        "explanation": "Views avoid duplicating memory and are essentially free, but they "
                        "create shared references that can be mutated unexpectedly by any "
                        "function that touches them -- as seen in Task 56. Explicit .copy() "
                        "guarantees isolation but costs memory and time proportional to the "
                        "array's size, which matters at millions of readings.",
        "reasoning": "A reasonable rule of thumb: rely on views for tightly-scoped, read-only "
                      "transformations inside a single function; copy defensively at function or "
                      "pipeline boundaries, where a caller shouldn't need to know or trust the "
                      "internal implementation of what they're calling.",
    },

    64: {
        "phase": "testing", "title": "When Is a Loop Actually Fine?",
        "hints": ["Think about operations where each row might need a genuinely different "
                   "amount of work, or where the fully vectorized version would barely be readable."],
        "answer_code": None,
        "explanation": "Task 51's per-row 'mean excluding zeros' is a good example: each row "
                        "can have a different count of valid (non-zero) values, making it "
                        "naturally ragged. Fully vectorizing that cleanly usually requires masked "
                        "arrays or notably less readable code, for a loop that only runs 12 times.",
        "reasoning": "The general principle: vectorize when it clearly wins on both "
                      "performance and clarity. When an array is small and the per-iteration "
                      "cost is dwarfed by how much clearer a small loop is to read and maintain, "
                      "choosing the loop is a legitimate engineering decision, not a shortcut.",
    },

    65: {
        "phase": "testing", "title": "flatten, ravel, or reshape(-1)?",
        "hints": ["Think about which of the three guarantee new memory versus which try to "
                   "avoid allocating it."],
        "answer_code": None,
        "explanation": ".flatten() always returns a copy -- new memory, always safe to mutate "
                        "independently. .ravel() returns a view whenever the data layout allows "
                        "it -- no copy, faster, but risks aliasing the source. .reshape(-1) "
                        "behaves like ravel internally (view when possible, copy when not); its "
                        "main purpose is general reshaping, with flattening to 1D as one special case.",
        "reasoning": "In a memory-constrained pipeline, prefer .ravel() or .reshape(-1) when "
                      "the result won't be mutated, to avoid unnecessary duplication; reach for "
                      ".flatten() specifically when you need a guaranteed independent copy.",
    },

    66: {
        "phase": "testing", "title": "Daily Machine Health Report",
        "hints": ["You've already computed each of these individually in earlier tasks -- "
                   "this asks you to assemble and order them together, and think critically "
                   "about relying on a single ranking metric."],
        "answer_code": '''avg_temp = np.nanmean(temperature, axis=1)
anomaly_counts = (vibration > 4.5).sum(axis=1)
total_production = production.sum(axis=1)

ranking = np.argsort(total_production)[::-1]

for idx in ranking:
    print(f"Machine {idx}: avg_temp={avg_temp[idx]:.1f}, anomalies={anomaly_counts[idx]}, "
          f"production={total_production[idx]}, downtime={downtime[idx]}")''',
        "explanation": "Each metric is computed with the axis-based/NaN-aware tools from "
                        "earlier tasks, then argsort on total_production (reversed) determines "
                        "the display order, best producer first.",
        "reasoning": "Ranking purely by production ignores downtime and anomaly counts -- a "
                      "high-output machine with heavy downtime or frequent faults may not "
                      "actually be the healthiest one. A composite, multi-criteria score (see "
                      "the Final Challenge) is generally more robust than a single metric.",
    },

    # ============================================================
    # FINAL CHALLENGE
    # ============================================================

    67: {
        "phase": "final", "title": "Factory Health Dashboard -- Integrated Challenge",
        "hints": ["This composes nearly every earlier task -- work through it in the five "
                   "numbered stages from the prompt rather than trying to solve it in one "
                   "expression, and reuse the cleaning logic from earlier hint-guided tasks."],
        "hint2": "Consider building a composite score from z-scores (value minus mean, "
                  "divided by std) of your chosen factors, so factors with different units and "
                  "ranges can be combined fairly.",
        "answer_code": '''# 1. Detect issues
nan_count = np.isnan(temperature).sum()
vib_anomalies = (vibration > 4.5).sum()
zero_pressure_faults = (pressure == 0).sum()
print("Missing temps:", nan_count, "| Vibration anomalies:", vib_anomalies,
      "| Zero-pressure faults:", zero_pressure_faults)

# 2. Clean each sensor array
temp_mean = np.nanmean(temperature)
temperature_clean = np.where(np.isnan(temperature), temp_mean, temperature)
vibration_clean = np.clip(vibration, 0, 5.0)
row_valid_means = np.array([
    pressure[i][pressure[i] != 0].mean() for i in range(pressure.shape[0])
])
pressure_clean = np.where(pressure == 0, row_valid_means.reshape(-1, 1), pressure)

# 3. Per-machine summary
avg_temp = temperature_clean.mean(axis=1)
avg_vibration = vibration_clean.mean(axis=1)
avg_pressure = pressure_clean.mean(axis=1)
total_production = production.sum(axis=1)

# 4. Composite health score (equal-weighted z-scores; lower vibration/downtime is better,
#    higher production is better)
def zscore(x):
    return (x - x.mean()) / x.std()

health_score = (
    zscore(total_production)
    - zscore(avg_vibration)
    - zscore(downtime)
)

# 5. Machine most in need of maintenance = lowest composite score
worst_idx = np.argmin(health_score)
print(f"Machine {worst_idx} needs maintenance: "
      f"production={total_production[worst_idx]}, "
      f"avg_vibration={avg_vibration[worst_idx]:.2f}, "
      f"downtime={downtime[worst_idx]}, "
      f"health_score={health_score[worst_idx]:.2f}")''',
        "explanation": "The pipeline detects issues first (counts only, no mutation), cleans "
                        "each array independently using the same techniques practiced earlier "
                        "(NaN fill, clipping, zero-fault correction), builds per-machine "
                        "summaries, then combines standardized (z-scored) factors into one "
                        "composite score so machines can be ranked fairly despite each factor "
                        "having very different units and scales.",
        "reasoning": "Equal weighting is a simplifying assumption -- in a real setting you'd "
                      "validate the weights against actual maintenance outcomes, and you'd want "
                      "domain input on which factors matter most. The z-score approach is "
                      "defensible because it prevents any one factor (e.g. production, with "
                      "large raw numbers) from dominating the score purely due to its scale.",
    },
}

_MIN_TASK = min(TASKS)
_MAX_TASK = max(TASKS)


def _check_task(n):
    if n not in TASKS:
        print(f"Task {n} doesn't exist. Valid task numbers are {_MIN_TASK}-{_MAX_TASK}.")
        return False
    return True


def hint(n, level=1):
    """Print a hint for task n. Use level=2 for a more direct hint where available."""
    if not _check_task(n):
        return
    task = TASKS[n]
    if level == 1:
        print(f"Hint for Task {n} — {task['title']}")
        print("-" * 40)
        print(task["hints"][0])
    elif level == 2:
        if "hint2" in task:
            print(f"Hint (level 2) for Task {n} — {task['title']}")
            print("-" * 40)
            print(task["hint2"])
        else:
            print(f"Task {n} only has one hint level. Try hint({n}) instead.")
    else:
        print(f"Invalid hint level: {level}. Supported levels are 1 and 2.")


def answer(n):
    """Reveal the full solution for task n -- including a live-computed preview of what it
    actually outputs, so you can check it against your own result."""
    if not _check_task(n):
        return
    task = TASKS[n]
    header = f"Task {n} — Solution"
    print(header)
    print("─" * len(header))
    print()
    if task["answer_code"]:
        print(task["answer_code"])
        print()
        output, error = _run_solution(task["answer_code"])
        if error:
            print(error)
            print()
        elif output:
            print("Expected Output:")
            print(output)
            print()
    print("Explanation:")
    print(task["explanation"])
    print()
    print("Key reasoning:")
    print(task["reasoning"])
