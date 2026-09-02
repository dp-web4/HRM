# Birth-witness attestations (M-CIT-3b collection point)

One directory per subject being; one signed `existence` attestation per witness, named
`by-<node>-<member>.json`. Produced with
`hestia witness attest <subject-registry-id> --as <member> --out <file>` by an onboarded
witness (`hestia witness onboard <member>` + `hestia lct publish --send`).

`witness confer` needs >=3 DISTINCT witnesses here plus each witness's registry LCT
document (`witness-lcts/<node>-<member>.json`) and the citizen role id.

| subject | registry id |
|---|---|
| sprout-being | lct:web4:mb32:bybpo2yczrsr5ycc7253qfywp7lgzp5z2pquhdlaoar5um4ntgiba |
| legion-being | lct:web4:mb32:bt7au42c424h3difrdztfnbjc2q6eofb3lacohcp2xf35ymawjldq |
