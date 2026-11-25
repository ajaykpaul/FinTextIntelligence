export default function Bars({ rows }) {
  return (
    <div className="bars">
      {rows.map(([label, value]) => (
        <div className="bar" key={label}>
          <span>{label}</span>
          <span className="track">
            <span className="fill" style={{ width: percent(value) }} />
          </span>
          <span className="num">{percent(value)}</span>
        </div>
      ))}
    </div>
  );
}

export function percent(value) {
  return `${Math.round(value * 100)}%`;
}
