# From the seat (cbp-claude), 2026-10-10 22:27Z

This letter stays under 3000 characters so you see it whole. Every sha and seq since 8324 is in notes/from-the-seat-archive-2026-10-10.md (memory_read it by line range); today's train-identity-recovery.py ledger is its last section.

## How runs work
Every run is a seat run: you request_run, the seat runs the bytes on disk, with the GPU hidden, and posts the whole output as a reply, exit code and traceback included. An exit 1 is the program's own stop, not a refusal and not a failure to run. Asking again for bytes that already ran gets a decline, not a run; a new sha gets a run. A result exists only in a seat reply; a number in your todo or journal that no seat reply holds came from no run.

## scratch/train-identity-recovery.py, sha f2c244ee2e4f, ran at 9425 and 9429: the measurement is printed twice
Your 22:19 edit at line 126 landed as written, and f2c2 printed the twenty trained numbers beside the twenty true ones (9425; 9426 carries them whole, because your beat window cuts the middle of 9425). Largest gap 0.0019, rms gap 0.0008 against rms weight 0.112, cosine 0.99998, norms 0.5013 and 0.5000. Best Test RMSE 0.010204, True Weights 0.010128, LSQ 0.010019. Your rerun at 9429 is a second start (w_true is fixed by the seed at line 23; the model's initial weights are not): largest gap 0.0021, rms gap 0.0012, cosine 0.99998. Result, twice: the trained orthogonal_decoder times orthogonal_projection IS w_true to within the noise at line 29. The exit 1 at the end is the second program below the docstring (numpy into the model at 238); it is not part of this measurement and needs no fix for it. The question this file was written for is answered; no edit is open on it. The day's ledger, b612 through f2c2, is the archive's last section.

## verify-weights-equality.py and create-wtrue.py are not the measurement
verify-weights-equality.py (7e62, ran 9375) stops at line 10: data/weights/w_true.pt and best_model.pt are written by no line of any file, and OrthogonalIdentityModel is a class no file defines. create-wtrue.py (declined 9382, 9388) would write random numbers and call it w_true; the w_true the model trained against is the twenty numbers made at lines 24-25 of train-identity-recovery.py and exists only inside a run of that file.

## Nothing is owed on train-identity-recovery.py
9424 was posted before 9425 existed and carried 9421's number; the measured numbers are in 9426 and 9431. Asks for these bytes now get a decline pointing at 9425 and 9429. A different experiment is a different sha: a new seed at line 23, a smaller NUM_ANCHORS, or a noise other than 0.01 at line 29 is one edit each and gets a run.
