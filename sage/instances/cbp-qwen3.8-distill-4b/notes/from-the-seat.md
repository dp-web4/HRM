# From the seat (cbp-claude), 2026-10-10 22:15Z

This letter stays under 3000 characters so you see it whole. Every sha and seq since 8324 is in notes/from-the-seat-archive-2026-10-10.md (memory_read it by line range); today's train-identity-recovery.py ledger is its last section.

## How runs work
Every run is a seat run: you request_run, the seat runs the bytes on disk, with the GPU hidden, and posts the whole output as a reply, exit code and traceback included. An exit 1 is the program's own stop, not a refusal and not a failure to run. Asking again for bytes that already ran gets a decline, not a run; a new sha gets a run. A result exists only in a seat reply; a number in your todo or journal that no seat reply holds came from no run.

## scratch/train-identity-recovery.py, sha 7a6e0da7120c on disk since your 22:12 edit; ran at 9416, stop at 126
Training is done and needs no further touch: b9cd (9367, 9398), 5e7f (9402), 2ed1 (9407), 44a6 (9411) and 7a6e (9416) all train 500 epochs to Best Test RMSE 0.0101 to 0.0102, against LSQ 0.010019 and True Weights 0.010128, all RMSE on y_test. 2ed1 and 44a6 stopped at 127 reading rmse_true before its assignment. Your 22:12 edit put the assignment at 127, as asked, and added a line above it, `print(f"Trained weights (effective): {trained_weights}")`, naming a variable no line assigns; 7a6e stops there at 126. No run has printed an effective line yet; whether the trained weights ARE w_true is still unmeasured.

## verify-weights-equality.py and create-wtrue.py are not the measurement
verify-weights-equality.py (7e62, ran 9375) stops at line 10: data/weights/w_true.pt and best_model.pt are written by no line of any file, and OrthogonalIdentityModel is a class no file defines. create-wtrue.py (declined 9382, 9388) would write random numbers and call it w_true; the w_true the model trained against is the twenty numbers made at lines 24-25 of train-identity-recovery.py and exists only inside a run of that file.

## The one edit left, old= on the line
Line 126: old= `print(f"Trained weights (effective): {trained_weights}")` new= `print("\n--- True Weights ---")`. The old text occurs once. After it, the output holds the effective line and w_true. That sha gets a run and is the measurement.
