const fs = require('fs');
const path = require('path');

const ocrBundlePath = path.join(
  __dirname,
  '..',
  'node_modules',
  '@paddlejs-models',
  'ocr',
  'lib',
  'index.js',
);

if (!fs.existsSync(ocrBundlePath)) {
  console.warn('[patch-paddle-ocr-leaks] OCR bundle not found, skipping.');
  process.exit(0);
}

const replacements = [
  {
    from: 'h().warpPerspective(o,D,i,n,h().INTER_CUBIC,h().BORDER_REPLICATE,new(h().Scalar));',
    to: '(function(A){h().warpPerspective(o,D,i,n,h().INTER_CUBIC,h().BORDER_REPLICATE,A),A.delete()})(new(h().Scalar));',
  },
  {
    from: 'h().warpAffine(D,t,F,s,h().INTER_CUBIC,h().BORDER_REPLICATE,new(h().Scalar))',
    to: '(function(A){h().warpAffine(D,t,F,s,h().INTER_CUBIC,h().BORDER_REPLICATE,A),A.delete()})(new(h().Scalar))',
  },
  {
    from: 'M.width=r.matSize[1],M.height=r.matSize[0],h().imshow(M,r),o.delete(),D.delete(),r.delete(),Q.delete(),E.delete()',
    to: 'M.width=r.matSize[1],M.height=r.matSize[0],h().imshow(M,r),o.delete(),D.delete(),r.delete(),Q.delete(),E.delete(),i.delete(),n.delete(),N.delete(),t&&(F.delete(),w.delete(),s.delete())',
  },
];

let bundleSource = fs.readFileSync(ocrBundlePath, 'utf8');
let changed = false;

for (const replacement of replacements) {
  if (bundleSource.includes(replacement.to)) {
    continue;
  }

  if (!bundleSource.includes(replacement.from)) {
    throw new Error(
      `[patch-paddle-ocr-leaks] Expected pattern not found: ${replacement.from.slice(0, 80)}...`,
    );
  }

  bundleSource = bundleSource.replace(replacement.from, replacement.to);
  changed = true;
}

if (changed) {
  fs.writeFileSync(ocrBundlePath, bundleSource);
  console.log('[patch-paddle-ocr-leaks] Applied OpenCV cleanup patch.');
} else {
  console.log('[patch-paddle-ocr-leaks] Patch already present.');
}
