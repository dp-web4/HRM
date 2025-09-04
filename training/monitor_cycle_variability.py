#!/usr/bin/env python3
"""
Monitor cycle variability during V2.3 training
"""
import subprocess
import re
import time

def get_latest_cycles():
    """Extract cycle count from latest training output"""
    try:
        # Get latest training output
        result = subprocess.run(['tail', '-20', '/dev/stderr'], 
                              capture_output=True, text=True, timeout=2)
        output = result.stderr
        
        # Find all cycle counts
        cycles = re.findall(r'cycles=(\d+\.?\d*)', output)
        halt_ps = re.findall(r'halt_p=(\d+\.?\d*)', output)
        
        if cycles and halt_ps:
            return float(cycles[-1]), float(halt_ps[-1])
        return None, None
    except:
        return None, None

def main():
    print("Monitoring V2.3 Cycle Variability")
    print("Expecting transition from 20 cycles → variable cycles")
    print("=" * 60)
    
    seen_cycles = set()
    last_cycles = None
    
    while True:
        cycles, halt_p = get_latest_cycles()
        
        if cycles is not None:
            seen_cycles.add(cycles)
            
            if cycles != last_cycles:
                print(f"Cycle count: {cycles}, Halt P: {halt_p:.3f}, "
                      f"Unique counts seen: {sorted(seen_cycles)}")
                last_cycles = cycles
            
            # Success condition
            if len(seen_cycles) > 3:
                print("\n🎉 SUCCESS: Multiple cycle counts detected!")
                print(f"Cycle variety: {sorted(seen_cycles)}")
                break
        
        time.sleep(2)

if __name__ == "__main__":
    main()