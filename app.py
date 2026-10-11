"""PathForge — FastAPI Backend.

Provides REST endpoints for pathfinding algorithms (BFS, DFS, Dijkstra, A*)
and serves the single-page frontend.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from algorithms import run_algorithm, run_all_algorithms

app = FastAPI(
    title="PathForge API",
    description="Shortest Path Visualizer & Algorithm Comparator for AI demonstrations.",
    version="1.0.0",
)

# Enable CORS for local testing flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)


# --- Request & Response Models ---

class CoordinateModel(BaseModel):
    row: int
    col: int


class RunRequest(BaseModel):
    rows: int = Field(default=20, ge=5, le=60, description="Grid height (rows)")
    cols: int = Field(default=35, ge=5, le=80, description="Grid width (cols)")
    start: List[int] = Field(..., description="[row, col] coordinates of start node")
    end: List[int] = Field(..., description="[row, col] coordinates of target node")
    walls: List[List[int]] = Field(default_factory=list, description="List of wall [row, col] coordinates")
    algorithm: str = Field(default="A*", description="Algorithm: 'BFS', 'DFS', 'Dijkstra', or 'A*'")

    @field_validator("start", "end")
    @classmethod
    def validate_coord(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("Coordinates must contain exactly [row, col].")
        return v


class CompareRequest(BaseModel):
    rows: int = Field(default=20, ge=5, le=60)
    cols: int = Field(default=35, ge=5, le=80)
    start: List[int]
    end: List[int]
    walls: List[List[int]] = Field(default_factory=list)

    @field_validator("start", "end")
    @classmethod
    def validate_coord(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError("Coordinates must contain exactly [row, col].")
        return v


# --- Helper Validation ---

def validate_grid_params(
    rows: int,
    cols: int,
    start: List[int],
    end: List[int],
    walls: List[List[int]],
) -> Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]:
    s_tuple = (start[0], start[1])
    e_tuple = (end[0], end[1])

    if not (0 <= s_tuple[0] < rows and 0 <= s_tuple[1] < cols):
        raise HTTPException(
            status_code=400,
            detail=f"Start node {start} is out of grid bounds ({rows}x{cols}).",
        )
    if not (0 <= e_tuple[0] < rows and 0 <= e_tuple[1] < cols):
        raise HTTPException(
            status_code=400,
            detail=f"Target node {end} is out of grid bounds ({rows}x{cols}).",
        )

    wall_tuples = [(w[0], w[1]) for w in walls if 0 <= w[0] < rows and 0 <= w[1] < cols]
    wall_set = set(wall_tuples)

    if s_tuple in wall_set:
        raise HTTPException(
            status_code=400,
            detail="Start node cannot be positioned on a wall cell.",
        )
    if e_tuple in wall_set:
        raise HTTPException(
            status_code=400,
            detail="Target node cannot be positioned on a wall cell.",
        )

    return s_tuple, e_tuple, wall_tuples


# --- Endpoints ---

@app.get("/")
def serve_index():
    """Serve the single-page visualizer frontend."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "project": "PathForge",
        "status": "ready",
        "message": "Frontend static/index.html will be placed in the next step.",
    }


@app.get("/api/info")
def get_algorithms_info() -> Dict[str, Any]:
    """Provide educational explanations and complexities for college AI demonstrations."""
    return {
        "BFS": {
            "name": "Breadth-First Search",
            "data_structure": "FIFO Queue",
            "guarantees_shortest_path": True,
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "description": "Explores concentric wave fronts. Guarantees the optimal shortest path on unweighted grids.",
        },
        "DFS": {
            "name": "Depth-First Search",
            "data_structure": "LIFO Stack",
            "guarantees_shortest_path": False,
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "description": "Explores deep into branches before backtracking. Fast but rarely produces the shortest path.",
        },
        "Dijkstra": {
            "name": "Dijkstra's Algorithm",
            "data_structure": "Min-Heap / Priority Queue",
            "guarantees_shortest_path": True,
            "time_complexity": "O((V + E) log V)",
            "space_complexity": "O(V)",
            "description": "Explores by lowest cumulative cost g(n). On unweighted grids, uniformly expands outward.",
        },
        "A*": {
            "name": "A* Search",
            "data_structure": "Min-Heap / Priority Queue",
            "guarantees_shortest_path": True,
            "time_complexity": "O(E)",
            "space_complexity": "O(V)",
            "description": "Informed search using f(n) = g(n) + h(n) with Manhattan distance heuristic. Minimizes explored nodes toward the target.",
        },
    }


@app.post("/api/run")
def run_single(req: RunRequest) -> Dict[str, Any]:
    """Execute a single search algorithm on the grid."""
    start_tuple, end_tuple, wall_tuples = validate_grid_params(
        req.rows, req.cols, req.start, req.end, req.walls
    )
    try:
        result = run_algorithm(
            algorithm=req.algorithm,
            rows=req.rows,
            cols=req.cols,
            start=start_tuple,
            end=end_tuple,
            walls=wall_tuples,
        )
        result["message"] = (
            f"Path found in {result['path_length']} steps."
            if result["success"]
            else "Target unreachable (no valid path exists)."
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(exc)}")


@app.post("/api/compare")
def compare_all(req: CompareRequest) -> Dict[str, Any]:
    """Execute BFS, DFS, Dijkstra, and A* simultaneously on the identical grid for side-by-side comparison."""
    start_tuple, end_tuple, wall_tuples = validate_grid_params(
        req.rows, req.cols, req.start, req.end, req.walls
    )
    try:
        results = run_all_algorithms(
            rows=req.rows,
            cols=req.cols,
            start=start_tuple,
            end=end_tuple,
            walls=wall_tuples,
        )

        # Build comparison summary
        summary = []
        for name, data in results.items():
            summary.append(
                {
                    "algorithm": name,
                    "success": data["success"],
                    "path_length": data["path_length"],
                    "nodes_explored": data["nodes_explored"],
                    "execution_time_ms": data["execution_time_ms"],
                }
            )

        return {
            "results": results,
            "summary": summary,
            "grid": {
                "rows": req.rows,
                "cols": req.cols,
                "start": req.start,
                "end": req.end,
                "wall_count": len(wall_tuples),
            },
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(exc)}")


# Serve static directory for static files (scripts, stylesheets, etc.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
