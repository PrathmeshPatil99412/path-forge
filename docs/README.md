# PathForge Documentation & Mini Project Assignment Sheet

This directory contains the academic submission documentation and individual assignment sheet for the **PathForge: Shortest Path Visualizer & Algorithm Comparator** mini project.

---

## 📄 Files in this Directory

| File | Format | Description |
| :--- | :--- | :--- |
| **[`mini_project_report.md`](file:///d:/ML%20projects/shortest-path-visualizer/docs/mini_project_report.md)** | Markdown (`.md`) | Comprehensive, academic technical report covering all 13 required sections (~3,800 words), complete with mathematical derivations, ASCII diagrams, real empirical benchmark tables, and IEEE references. |
| **[`mini_project_report.html`](file:///d:/ML%20projects/shortest-path-visualizer/docs/mini_project_report.html)** | HTML (`.html`) | Print-ready, professionally styled document formatted specifically for A4 PDF conversion. Includes institutional assignment header, student particulars grid, evaluator signature block, page breaks, and `@media print` rules. |

---

## 🖨️ How to Export to PDF (7–8 Pages)

To generate the final PDF for submission:

1. Open **[`docs/mini_project_report.html`](file:///d:/ML%20projects/shortest-path-visualizer/docs/mini_project_report.html)** in any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, or Brave).
2. Click the **"🖨️ Print / Save as PDF (A4)"** button at the top (or press `Ctrl + P`).
3. In the print dialog settings:
   - **Destination:** Save as PDF / Microsoft Print to PDF
   - **Paper Size:** A4
   - **Margins:** Default or Minimum
   - **Options:** Enable *"Background graphics"* (to render table header backgrounds and badges)
4. Click **Save** to export the clean, 7–8 page individual assignment sheet.

---

## 📋 Required Report Sections Included

All 13 required institutional sections are fully covered:

1. **Title of the project**
2. **Problem statement** (Mathematical configuration space formulation, 4-connected grid graph model)
3. **Objectives** (Algorithmic implementation, heuristic formulation, dynamic visualization, benchmarking)
4. **Dataset/Input description** (Implicit 2D grid matrix, obstacle hash sets, 4 standard benchmark topologies)
5. **Algorithm/AI-ML technique used** (BFS, DFS, Dijkstra, A*, Manhattan distance heuristic, proofs of admissibility and consistency)
6. **Methodology/approach** (Decoupled client-server architecture, min-heap tie-breaking invariant, backtracking)
7. **Implementation details** (FastAPI backend, Pydantic data models, Vanilla HTML/CSS/JS frontend)
8. **Results/output** (Empirical benchmark scenarios and wavefront progression)
9. **Performance evaluation** (Quantitative metrics: search space reduction %, exploration efficiency ratio $\eta$, suboptimality ratio $\rho$)
10. **Analysis/discussion** ($L_1$ vs $L_2$ vs $L_\infty$ distance metrics, U-trap local minima resilience, edge case handling)
11. **Conclusion** (Summary of algorithmic performance and findings)
12. **Future scope** (8-directional movement, dynamic D* Lite, continuous RRT*, multi-agent CBS)
13. **References** (12 formal IEEE/academic citations)

---

## 🧪 Re-running the Benchmarks

To verify the test suite or re-generate benchmark metrics:

```bash
# Run unit tests
python -m unittest test_algorithms.py test_app.py
```
