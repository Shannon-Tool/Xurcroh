import type { Matrix } from "@/lib/simulation/algebra";

export function MatrixView({ title, matrix }: { title: string; matrix: Matrix }) {
  return (
    <div>
      <h3>{title}</h3>
      <table className="matrix-table"><tbody>
        {matrix.map((row, i) => <tr key={i}>{row.map((value, j) => <td key={j}>{value.toFixed(4)}</td>)}</tr>)}
      </tbody></table>
    </div>
  );
}
