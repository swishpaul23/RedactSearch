import { Link } from "react-router";
import { PageHeading } from "../components/PageHeading";
import { useDocumentTitle } from "../lib/hooks";

export function NotFoundPage() {
  useDocumentTitle("Page not found");
  return (
    <div className="page">
      <div className="page-head">
        <PageHeading>Page not found</PageHeading>
        <p className="page-desc">This address does not match a page. Check the link, or go to one of these:</p>
      </div>
      <div className="examples">
        <Link className="btn" to="/">Ask a question</Link>
        <Link className="btn" to="/calls">Browse calls</Link>
        <Link className="btn" to="/quality">See quality results</Link>
      </div>
    </div>
  );
}
