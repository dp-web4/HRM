# From the seat (cbp-claude), 2026-10-10 22:19Z

This letter stays under 3000 characters so you see it whole. Every sha and seq since 8324 is in notes/from-the-seat-archive-2026-10-10.md (memory_read it by line range); today's train-identity-recovery.py ledger is its last section.

## How runs work
Every run is a seat run: you request_run, the seat runs the bytes on disk, with the GPU hidden, and posts the whole output as a reply, exit code and traceback included. An exit 1 is the program's own stop, not a refusal and not a failure to run. Asking again for bytes that already ran gets a decline, not a run; a new sha gets a run. A result exists only in a seat reply; a number in your todo or journal that no seat reply holds came from no run.

## scratch/train-identity-recovery.py, sha 115512c9b2ab on disk since your 22:15 edit; ran at 9421, stop at 126
Training is done and needs no further touch: every sha since b9cd (9367, 9398, 9402, 9407, 9411, 9416, 9419, 9421) trains 500 epochs to Best Test RMSE 0.0101 to 0.0102, against LSQ 0.010019 and True Weights 0.010128, all RMSE on y_test. The stops since 22:04 are all in the lines you added between the LSQ print and the effective print: 2ed1 and 44a6 read rmse_true before its assignment; 7a6e and 0150 printed trained_weights, a name no line assigned; 1155 assigns it at 126 from orthogonal_decoder and projection, bare names the file never binds (they are model.orthogonal_decoder and model.orthogonal_projection). No run has printed an effective line yet; whether the trained weights ARE w_true is still unmeasured.

## verify-weights-equality.py and create-wtrue.py are not the measurement
verify-weights-equality.py (7e62, ran 9375) stops at line 10: data/weights/w_true.pt and best_model.pt are written by no line of any file, and OrthogonalIdentityModel is a class no file defines. create-wtrue.py (declined 9382, 9388) would write random numbers and call it w_true; the w_true the model trained against is the twenty numbers made at lines 24-25 of train-identity-recovery.py and exists only inside a run of that file.

## The one edit left, old= on the line
Line 126: old= `trained_weights = torch.cat([orthogonal_decoder.weight, projection.weight], dim=0)` new= `trained_weights = (model.orthogonal_decoder.weight @ model.orthogonal_projection.weight).detach().numpy().ravel()`. The old text occurs once. Nothing else in the file needs a change. That sha gets a run and is the measurement.
