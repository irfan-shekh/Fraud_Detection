"""
Submission Archive Packager for BIA Fraud Detection Capstone.
Creates Fraud_Detection_BIA_Final_Submission.zip containing the required deliverables
and verifies archive contents.
"""

import os
import zipfile


def create_submission_zip(zip_name: str = "Fraud_Detection_BIA_Final_Submission.zip"):
    print(f"[ZIP] Creating BIA submission package: '{zip_name}'...")

    files_to_include = [
        ("notebooks/Fraud_Detection_Analysis.ipynb", "Fraud_Detection_Analysis.ipynb"),
        ("reports/Fraud_Detection_Final_Report.docx", "Fraud_Detection_Final_Report.docx"),
        ("reports/Fraud_Detection_Presentation.pptx", "Fraud_Detection_Presentation.pptx"),
        ("README.md", "README.md"),
        ("requirements.txt", "requirements.txt"),
        ("reports/model_comparison.csv", "reports/model_comparison.csv"),
        ("reports/evaluation_report.json", "reports/evaluation_report.json"),
        ("reports/financial_impact_analysis.json", "reports/financial_impact_analysis.json"),
        ("reports/data_quality_report.json", "reports/data_quality_report.json"),
        ("reports/FINAL_INDEPENDENT_AUDIT.md", "reports/FINAL_INDEPENDENT_AUDIT.md"),
        ("reports/BIA_CAPSTONE_VIVA_PREPARATION.md", "reports/BIA_CAPSTONE_VIVA_PREPARATION.md"),
        ("reports/BI_DASHBOARD_GUIDE.md", "reports/BI_DASHBOARD_GUIDE.md"),
        ("reports/fraud_detection_bi_dataset.csv", "reports/fraud_detection_bi_dataset.csv"),
    ]

    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add primary deliverables
        for src_path, arc_name in files_to_include:
            if os.path.exists(src_path):
                zipf.write(src_path, arc_name)
                print(f"  Added: {arc_name}")
            else:
                print(f"  Warning: {src_path} not found!")

        # Add figures
        figures_dir = "reports/figures"
        if os.path.exists(figures_dir):
            for fig in os.listdir(figures_dir):
                if fig.endswith(".png"):
                    fig_src = os.path.join(figures_dir, fig)
                    fig_arc = os.path.join("reports/figures", fig)
                    zipf.write(fig_src, fig_arc)
            print("  Added all diagnostic figures from reports/figures/")

        # Add core source code (src/ and dashboard/)
        for folder in ["src", "dashboard", "tests"]:
            if os.path.exists(folder):
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.endswith((".py", ".json")):
                            src_file = os.path.join(root, file)
                            arc_file = os.path.relpath(src_file, ".")
                            zipf.write(src_file, arc_file)
                print(f"  Added source package: {folder}/")

        if os.path.exists("run_project.py"):
            zipf.write("run_project.py", "run_project.py")
            print("  Added: run_project.py")

    file_size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    print(f"[ZIP] Archive successfully generated: {zip_name} ({file_size_mb:.2f} MB)")

    # Verify zip integrity
    with zipfile.ZipFile(zip_name, "r") as zipf:
        namelist = zipf.namelist()
        test_res = zipf.testzip()
        if test_res is None:
            print(f"[ZIP] Archive verification PASSED. Total items: {len(namelist)}")
        else:
            print(f"[ZIP] Archive verification FAILED at: {test_res}")

    return zip_name


if __name__ == "__main__":
    create_submission_zip()

