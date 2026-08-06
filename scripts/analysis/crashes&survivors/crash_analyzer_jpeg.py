import sys
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_execution_logs(root_directory):
    directory = Path(root_directory)

    # Counters
    total_images_analyzed = 0
    flow_timeouts = 0      # Empty files (0 bytes)
    data_crashes = 0       # Files containing "segfault" in the middle of the binary
    successes = 0          # Survived (image might be good or bad, but didn't crash)

    print(f"--- Starting scan in: {directory.resolve()} ---")

    # Target extensions to analyze (add others if needed, like .png)
    target_extensions = {'.jpg', '.jpeg', '.tif', '.tiff'}

    # Recursively search for all files
    for file in directory.rglob("*"):
        if file.suffix.lower() not in target_extensions:
            continue

        total_images_analyzed += 1

        # 1. Check Empty Files (0 bytes) -> Immediate Timeout / Flow Crash
        if file.stat().st_size == 0:
            flow_timeouts += 1
            continue

        try:
            # Open IMAGE file as text, ignoring binary characters
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()

                # 2. Check Memory Failure -> Data Crash (Segfault) hidden in JPEG
                if "segfault" in content and "pc" in content and "va/inst" in content:
                    data_crashes += 1

                # 3. If not empty and no segfault -> Intact Execution
                else:
                    successes += 1

        except Exception as e:
            print(f"[Error] Failed to read file {file.name}: {e}")

    # --- Final Report ---
    if total_images_analyzed == 0:
        print("\nNo image files found for analysis!")
        return

    total_failures = data_crashes + flow_timeouts
    error_percentage = (total_failures / total_images_analyzed) * 100 if total_images_analyzed > 0 else 0

    print("\n" + "="*55)
    print("            FAULT INJECTION REPORT (CRASHES)")
    print("="*55)
    print(f" Total Images Analyzed    : {total_images_analyzed}")
    print("-" * 55)
    print(f" [Data Crash] (Segfault)  : {data_crashes}")
    print(f" [Flow/Timeout] (0 bytes) : {flow_timeouts}")
    print(f" Successful Executions    : {successes}")
    print("="*55)
    print(f" TOTAL CRASH RATE (Fatal) : {error_percentage:.2f}%")
    print("="*55)


    # ==========================================
    # GRAPH GENERATION (MATPLOTLIB)
    # ==========================================
    categories = ['Data Crash\n(Segfault)', 'Flow / Timeout\n(0 bytes)', 'Successes\n(Survivors)']
    values = [data_crashes, flow_timeouts, successes]
    colors = ['#8B0000', '#A9A9A9', '#2E8B57'] # Red, Gray, Green

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, values, color=colors)

    # Place numbers on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (total_images_analyzed * 0.01),
                 f'{height}', ha='center', va='bottom', fontweight='bold')

    plt.title(f'Crash Distribution - {directory.name}\nFailure Rate: {error_percentage:.2f}%', fontsize=14)
    plt.ylabel('Number of Images')
    plt.ylim(0, total_images_analyzed * 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save PNG image in the same folder where you ran the command
    graph_name = f"crash_report_{directory.name}.png"
    plt.savefig(graph_name, bbox_inches='tight')
    print(f"\n[+] Graph saved successfully: {graph_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 crash_analyzer.py <directory>")
    else:
        analyze_execution_logs(sys.argv[1])
