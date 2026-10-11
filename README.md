# ⚡ PathForge — Shortest Path Visualizer & Algorithm Comparator

**PathForge** is a full-featured, lightweight AI mini-project built for demonstrating and comparing classic graph search algorithms on 2D grids. It connects a Python + FastAPI backend with a modern, responsive vanilla JavaScript/CSS interactive visualizer.

---

## 🌟 Key Features

1. **Interactive Grid Playground**:
   - Click and drag anywhere to draw or erase obstacle walls.
   - Drag and drop Start (`S`) and Target (`T`) nodes to relocate them dynamically.
   - Built-in maze presets: **Random Obstacles (30%)**, **U-Trap Pocket**, **Narrow Corridors**, and **Concentric Spiral**.

2. **Core AI Search Algorithms (`algorithms.py`)**:
   - **A\* Search**: Informed search using the **Manhattan distance** heuristic on a 4-directional grid ($h(n) = |\Delta r| + |\Delta c|$). Guarantees the shortest path with targeted node exploration.
   - **Dijkstra's Algorithm**: Min-heap uniform-cost search. Guarantees the shortest path by radially expanding from the source.
   - **Breadth-First Search (BFS)**: Level-order search using a FIFO queue. Guarantees the optimal path on unweighted graphs.
   - **Depth-First Search (DFS)**: Uninformed search using a LIFO stack. Explores deep branches first; does not guarantee the shortest path.

3. **Algorithm Comparator**:
   - Execute all 4 algorithms concurrently on the identical maze with one click (`Compare All 4`).
   - Benchmarks and displays:
     - **Nodes Explored** (search efficiency)
     - **Path Length** (optimality)
     - **Backend Execution Time** (milliseconds)
     - **Optimality Verdict** (Optimal vs Suboptimal / Failed)

4. **Dynamic Visualizer Controls**:
   - Dual-phase smooth animation: Explored expansion wavefront $\rightarrow$ Shortest path draw.
   - Real-time animation speed slider (1ms to 50ms, plus **Instant** mode).
   - Clear Path, Clear Walls, and Full Reset controls.

5. **College AI Practical & Viva Reference**:
   - In-app algorithm theory cards explaining evaluation functions ($f(n) = g(n) + h(n)$), heuristics, data structures, and time/space complexity.

---

## 📁 Project Structure

```text
shortest-path-visualizer/
├── app.py              # FastAPI application, REST endpoints, and static file serving
├── algorithms.py       # BFS, DFS, Dijkstra, and A* search logic & metrics
├── docs/               # Mini Project Report (Markdown & Print-Ready HTML Assignment Sheet)
│   ├── mini_project_report.md
│   ├── mini_project_report.html
│   └── README.md
├── static/
│   └── index.html      # Responsive visualizer UI (HTML5, Vanilla CSS, Vanilla JS)
├── test_algorithms.py  # Unit tests for algorithm correctness & path optimality
├── test_app.py         # Unit tests for FastAPI endpoints & request validation
├── requirements.txt    # Minimal dependencies (fastapi, uvicorn, pydantic)
└── README.md           # Documentation & presentation guide
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the Application
Run the FastAPI development server:
```bash
python -m uvicorn app:app --reload --port 8000
```
or directly:
```bash
python app.py
```

### 4. Open in Browser
Navigate to:
```text
http://127.0.0.1:8000
```

Interactive API documentation (Swagger UI) is also available at:
```text
http://127.0.0.1:8000/docs
```

---

## 🧪 Running the Tests

To verify algorithm optimality, Manhattan distance calculations, and API route handlers:

```bash
# Test algorithms (optimality, bounds, obstacles)
python -m unittest test_algorithms.py

# Test API endpoints (validation, unreachable targets, comparison)
python -m unittest test_app.py
```

---

## 📊 Algorithm Comparison Summary

| Algorithm | Data Structure | Heuristic | Guarantees Shortest Path? | Time Complexity | Space Complexity |
|---|---|---|---|---|---|
| **A\*** | Min-Heap / Priority Queue | Manhattan Distance | **Yes** (Admissible) | $\mathcal{O}(E)$ | $\mathcal{O}(V)$ |
| **Dijkstra** | Min-Heap / Priority Queue | None ($h=0$) | **Yes** | $\mathcal{O}((V + E)\log V)$ | $\mathcal{O}(V)$ |
| **BFS** | FIFO Queue | None | **Yes** (Unweighted) | $\mathcal{O}(V + E)$ | $\mathcal{O}(V)$ |
| **DFS** | LIFO Stack | None | **No** | $\mathcal{O}(V + E)$ | $\mathcal{O}(V)$ |

---

## 🎯 Viva / College Demo Tips

1. **Why does A\* explore fewer nodes than BFS and Dijkstra on an open grid?**
   - *Answer*: BFS and Dijkstra expand uniformly in all directions (like an expanding circle/diamond). A\* adds the Manhattan distance heuristic $h(n)$ toward the target node, steering search exploration toward the goal.
2. **Why is Manhattan distance used instead of Euclidean distance here?**
   - *Answer*: Movement is strictly 4-directional (Up, Down, Left, Right). Manhattan distance represents the exact minimum grid steps without diagonal movement, making it admissible ($h(n) \le h^*(n)$) and consistent.
3. **When does DFS fail to find the shortest path?**
   - *Answer*: DFS dives deep down whichever branch it explores first (e.g. going all the way down and around obstacles) until it hits the target or a dead end. It does not consider edge count or cumulative path distance.
