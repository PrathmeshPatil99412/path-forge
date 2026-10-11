"""PathForge Search Algorithms Module.

Provides implementations for 4-directional grid pathfinding:
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- Dijkstra's Algorithm
- A* Search (Manhattan Distance)

Each algorithm tracks explored order, reconstructs the final path,
and measures performance metrics (execution time, nodes visited, path length).
"""

from collections import deque
import heapq
import time
from typing import Any, Dict, List, Optional, Set, Tuple

Coordinate = Tuple[int, int]

# 4 cardinal directions: Up, Right, Down, Left
DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def manhattan_distance(p1: Coordinate, p2: Coordinate) -> int:
    """Calculate Manhattan distance between two points on a 4-directional grid."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def is_valid_cell(
    cell: Coordinate, rows: int, cols: int, wall_set: Set[Coordinate]
) -> bool:
    """Check if a coordinate is within bounds and not an obstacle."""
    r, c = cell
    return 0 <= r < rows and 0 <= c < cols and cell not in wall_set


def reconstruct_path(
    parent_map: Dict[Coordinate, Optional[Coordinate]], end: Coordinate
) -> List[Coordinate]:
    """Reconstruct path from start to end using the parent map."""
    path = []
    curr: Optional[Coordinate] = end
    while curr is not None:
        path.append(curr)
        curr = parent_map.get(curr)
    path.reverse()
    return path


def bfs(
    rows: int,
    cols: int,
    start: Coordinate,
    end: Coordinate,
    wall_set: Set[Coordinate],
) -> Dict[str, Any]:
    """Breadth-First Search (BFS) - Guarantees shortest path on unit-cost grid."""
    start_time = time.perf_counter()

    if start == end:
        return {
            "algorithm": "BFS",
            "success": True,
            "path": [list(start)],
            "explored": [list(start)],
            "path_length": 0,
            "nodes_explored": 1,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }

    queue: deque = deque([start])
    visited: Set[Coordinate] = {start}
    parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}
    explored_order: List[List[int]] = []
    found = False

    while queue:
        current = queue.popleft()
        explored_order.append(list(current))

        if current == end:
            found = True
            break

        for dr, dc in DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)
            if (
                is_valid_cell(neighbor, rows, cols, wall_set)
                and neighbor not in visited
            ):
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    exec_time = round((time.perf_counter() - start_time) * 1000, 3)

    if found:
        path_coords = reconstruct_path(parent, end)
        return {
            "algorithm": "BFS",
            "success": True,
            "path": [list(p) for p in path_coords],
            "explored": explored_order,
            "path_length": max(0, len(path_coords) - 1),
            "nodes_explored": len(explored_order),
            "execution_time_ms": exec_time,
        }

    return {
        "algorithm": "BFS",
        "success": False,
        "path": [],
        "explored": explored_order,
        "path_length": 0,
        "nodes_explored": len(explored_order),
        "execution_time_ms": exec_time,
    }


def dfs(
    rows: int,
    cols: int,
    start: Coordinate,
    end: Coordinate,
    wall_set: Set[Coordinate],
) -> Dict[str, Any]:
    """Depth-First Search (DFS) - Explores deeply, not guaranteed shortest path."""
    start_time = time.perf_counter()

    if start == end:
        return {
            "algorithm": "DFS",
            "success": True,
            "path": [list(start)],
            "explored": [list(start)],
            "path_length": 0,
            "nodes_explored": 1,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }

    stack: List[Coordinate] = [start]
    visited: Set[Coordinate] = set()
    parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}
    explored_order: List[List[int]] = []
    found = False

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        explored_order.append(list(current))

        if current == end:
            found = True
            break

        # Process neighbors (reverse direction to maintain intuitive order on stack)
        for dr, dc in reversed(DIRECTIONS):
            neighbor = (current[0] + dr, current[1] + dc)
            if (
                is_valid_cell(neighbor, rows, cols, wall_set)
                and neighbor not in visited
            ):
                if neighbor not in parent:
                    parent[neighbor] = current
                stack.append(neighbor)

    exec_time = round((time.perf_counter() - start_time) * 1000, 3)

    if found:
        path_coords = reconstruct_path(parent, end)
        return {
            "algorithm": "DFS",
            "success": True,
            "path": [list(p) for p in path_coords],
            "explored": explored_order,
            "path_length": max(0, len(path_coords) - 1),
            "nodes_explored": len(explored_order),
            "execution_time_ms": exec_time,
        }

    return {
        "algorithm": "DFS",
        "success": False,
        "path": [],
        "explored": explored_order,
        "path_length": 0,
        "nodes_explored": len(explored_order),
        "execution_time_ms": exec_time,
    }


def dijkstra(
    rows: int,
    cols: int,
    start: Coordinate,
    end: Coordinate,
    wall_set: Set[Coordinate],
) -> Dict[str, Any]:
    """Dijkstra's Algorithm - Guarantees shortest path (uniform edge cost = 1)."""
    start_time = time.perf_counter()

    if start == end:
        return {
            "algorithm": "Dijkstra",
            "success": True,
            "path": [list(start)],
            "explored": [list(start)],
            "path_length": 0,
            "nodes_explored": 1,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }

    # Heap elements: (distance, insertion_counter, coordinate)
    counter = 0
    pq: List[Tuple[int, int, Coordinate]] = [(0, counter, start)]
    distances: Dict[Coordinate, int] = {start: 0}
    parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}
    visited: Set[Coordinate] = set()
    explored_order: List[List[int]] = []
    found = False

    while pq:
        dist, _, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)
        explored_order.append(list(current))

        if current == end:
            found = True
            break

        for dr, dc in DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not is_valid_cell(neighbor, rows, cols, wall_set):
                continue

            new_dist = dist + 1
            if new_dist < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_dist
                parent[neighbor] = current
                counter += 1
                heapq.heappush(pq, (new_dist, counter, neighbor))

    exec_time = round((time.perf_counter() - start_time) * 1000, 3)

    if found:
        path_coords = reconstruct_path(parent, end)
        return {
            "algorithm": "Dijkstra",
            "success": True,
            "path": [list(p) for p in path_coords],
            "explored": explored_order,
            "path_length": max(0, len(path_coords) - 1),
            "nodes_explored": len(explored_order),
            "execution_time_ms": exec_time,
        }

    return {
        "algorithm": "Dijkstra",
        "success": False,
        "path": [],
        "explored": explored_order,
        "path_length": 0,
        "nodes_explored": len(explored_order),
        "execution_time_ms": exec_time,
    }


def astar(
    rows: int,
    cols: int,
    start: Coordinate,
    end: Coordinate,
    wall_set: Set[Coordinate],
) -> Dict[str, Any]:
    """A* Search - Guarantees shortest path using Manhattan distance heuristic."""
    start_time = time.perf_counter()

    if start == end:
        return {
            "algorithm": "A*",
            "success": True,
            "path": [list(start)],
            "explored": [list(start)],
            "path_length": 0,
            "nodes_explored": 1,
            "execution_time_ms": round((time.perf_counter() - start_time) * 1000, 3),
        }

    counter = 0
    h_start = manhattan_distance(start, end)
    # Heap tuple: (f_cost, h_cost, counter, coordinate)
    pq: List[Tuple[int, int, int, Coordinate]] = [(h_start, h_start, counter, start)]
    g_costs: Dict[Coordinate, int] = {start: 0}
    parent: Dict[Coordinate, Optional[Coordinate]] = {start: None}
    visited: Set[Coordinate] = set()
    explored_order: List[List[int]] = []
    found = False

    while pq:
        _, _, _, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)
        explored_order.append(list(current))

        if current == end:
            found = True
            break

        current_g = g_costs[current]

        for dr, dc in DIRECTIONS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not is_valid_cell(neighbor, rows, cols, wall_set):
                continue

            tentative_g = current_g + 1
            if tentative_g < g_costs.get(neighbor, float("inf")):
                g_costs[neighbor] = tentative_g
                h = manhattan_distance(neighbor, end)
                f = tentative_g + h
                parent[neighbor] = current
                counter += 1
                heapq.heappush(pq, (f, h, counter, neighbor))

    exec_time = round((time.perf_counter() - start_time) * 1000, 3)

    if found:
        path_coords = reconstruct_path(parent, end)
        return {
            "algorithm": "A*",
            "success": True,
            "path": [list(p) for p in path_coords],
            "explored": explored_order,
            "path_length": max(0, len(path_coords) - 1),
            "nodes_explored": len(explored_order),
            "execution_time_ms": exec_time,
        }

    return {
        "algorithm": "A*",
        "success": False,
        "path": [],
        "explored": explored_order,
        "path_length": 0,
        "nodes_explored": len(explored_order),
        "execution_time_ms": exec_time,
    }


ALGORITHM_MAP = {
    "bfs": bfs,
    "dfs": dfs,
    "dijkstra": dijkstra,
    "astar": astar,
    "a*": astar,
}


def run_algorithm(
    algorithm: str,
    rows: int,
    cols: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    walls: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Run a specified algorithm on the given grid configuration."""
    key = algorithm.strip().lower()
    if key not in ALGORITHM_MAP:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Choose from: {list(ALGORITHM_MAP.keys())}"
        )

    # Basic validations
    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        raise ValueError(f"Start node {start} is outside the grid bounds ({rows}x{cols})")
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        raise ValueError(f"End node {end} is outside the grid bounds ({rows}x{cols})")

    wall_set = {tuple(w) for w in walls}
    if start in wall_set:
        raise ValueError("Start node cannot be placed on a wall")
    if end in wall_set:
        raise ValueError("End node cannot be placed on a wall")

    fn = ALGORITHM_MAP[key]
    return fn(rows, cols, start, end, wall_set)


def run_all_algorithms(
    rows: int,
    cols: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    walls: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Execute BFS, DFS, Dijkstra, and A* on the identical grid for direct comparison."""
    results = {}
    for name in ["BFS", "DFS", "Dijkstra", "A*"]:
        results[name] = run_algorithm(name, rows, cols, start, end, walls)
    return results
