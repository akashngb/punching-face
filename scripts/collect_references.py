"""Discover public impact-reference pages, keeping provenance and review state.

Search results are candidates, never an automatically labelled training set.
No image files are downloaded and no captured user data is transmitted.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode, urlparse, quote
import xml.etree.ElementTree as ET
import json, datetime, argparse

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ['"punch" "face" slow motion boxing', '"boxing" "cheek" impact deformation']


def collect(queries=QUERIES):
    folder = ROOT / '.local/impact-references'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'catalog.json'
    catalog = json.loads(path.read_text()) if path.exists() else {'references': []}
    found = {r.get('url') for r in catalog['references']}
    runs = []
    for query in queries:
        url = 'https://www.bing.com/search?' + urlencode(
            {'q': query, 'format': 'rss'}, quote_via=quote
        )
        run = {
            'query': query,
            'provider': 'Bing RSS',
            'searchedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'added': 0,
            'rejected': 0,
        }
        try:
            raw = urlopen(
                Request(url, headers={'User-Agent': 'ContactReferenceResearch/0.1'}),
                timeout=15,
            ).read(2_000_000)
            root = ET.fromstring(raw)
            for item in root.findall('.//item')[:12]:
                link = item.findtext('link', '')
                if urlparse(link).scheme not in ('http', 'https') or link in found:
                    continue
                content = (
                    item.findtext('title', '') + ' ' + item.findtext('description', '')
                ).lower()
                if not any(
                    t in content for t in ('punch', 'boxing', 'glove', 'impact')
                ) or not any(t in content for t in ('face', 'facial', 'cheek', 'skin')):
                    run['rejected'] += 1
                    continue
                found.add(link)
                run['added'] += 1
                catalog['references'].append(
                    {
                        'id': f'web-{len(found)}',
                        'title': item.findtext('title', ''),
                        'url': link,
                        'description': item.findtext('description', ''),
                        'query': query,
                        'status': 'candidate: relevance and rights unverified',
                        'license': 'unknown',
                        'features': [],
                    }
                )
        except Exception as exc:
            run['error'] = str(exc)
        runs.append(run)
    catalog.setdefault('searchRuns', []).extend(runs)
    path.write_text(json.dumps(catalog, indent=2))
    return catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', action='append')
    args = parser.parse_args()
    data = collect(args.query or QUERIES)
    print(json.dumps(data['searchRuns'][-len(args.query or QUERIES) :], indent=2))
