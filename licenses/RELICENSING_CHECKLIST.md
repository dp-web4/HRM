# Relicensing Checklist: Apache 2.0 → AGPLv3

Follow these steps to properly relicense a derivative of Apache 2.0 code under AGPLv3:

1. **Keep Apache Notices**
   - Include the original `LICENSE-APACHE` file (Apache 2.0 text).
   - Preserve all original copyright headers in source files.

2. **Add AGPLv3 License**
   - Create a new file `LICENSE-AGPL` containing the full AGPLv3 text.
   - Clearly state in your README that the project as a whole is licensed under AGPLv3.

3. **Add a NOTICE File**
   - Document original authorship and your modifications in `NOTICE`.
   - Reference both Apache 2.0 and AGPLv3.

4. **Update Headers (Optional but Recommended)**
   - At the top of modified source files, add:
     ```
     This file has been modified by [Your Name/Org] and is licensed under the AGPLv3.
     ```

5. **Network Use Requirement**
   - If users access your software over a network (e.g. web app, service), you must provide a clear link or method to download the complete corresponding source code.

6. **Binary Distribution**
   - If you distribute binaries, provide the source code or an offer to obtain it.

---

## Directory Layout Example

```
project/
  licenses/
    LICENSE-APACHE
    LICENSE-AGPL
    NOTICE
    RELICENSING_CHECKLIST.md
```

