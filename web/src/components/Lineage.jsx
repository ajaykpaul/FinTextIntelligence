export default function Lineage({ result }) {
  return (
    <details>
      <summary>Lineage / raw JSON</summary>
      <dl className="lineage">
        {Object.entries(result.lineage).map(([field, value]) => (
          <div key={field}>
            <dt>{field.replace(/_/g, " ")}</dt>
            <dd>{typeof value === "object" ? JSON.stringify(value) : value}</dd>
          </div>
        ))}
      </dl>
      <pre>{JSON.stringify(result, null, 2)}</pre>
    </details>
  );
}
