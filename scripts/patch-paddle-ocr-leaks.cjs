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

const scalarRepairs = [
  {
    from: '(function(A){h().warpPerspective(o,D,i,n,h().INTER_CUBIC,h().BORDER_REPLICATE,A),A.delete()})(new(h().Scalar));',
    to: 'h().warpPerspective(o,D,i,n,h().INTER_CUBIC,h().BORDER_REPLICATE,new(h().Scalar));',
  },
  {
    from: '(function(A){h().warpAffine(D,t,F,s,h().INTER_CUBIC,h().BORDER_REPLICATE,A),A.delete()})(new(h().Scalar))',
    to: 'h().warpAffine(D,t,F,s,h().INTER_CUBIC,h().BORDER_REPLICATE,new(h().Scalar))',
  },
];

const guardedCleanupExpression =
  'M.width=r.matSize[1],M.height=r.matSize[0],h().imshow(M,r),o.delete(),D.delete(),r.delete(),Q.delete(),E.delete(),i&&i.delete&&i.delete(),n&&n.delete&&n.delete(),N&&N.delete&&N.delete(),t&&(F&&F.delete&&F.delete(),w&&w.delete&&w.delete(),s&&s.delete&&s.delete())';

const cleanupRepairs = [
  {
    from: 'M.width=r.matSize[1],M.height=r.matSize[0],h().imshow(M,r),o.delete(),D.delete(),r.delete(),Q.delete(),E.delete(),i.delete(),n.delete(),N.delete(),t&&(F.delete(),w.delete(),s.delete())',
    to: guardedCleanupExpression,
  },
];

const cleanupPatches = [
  {
    from: 'M.width=r.matSize[1],M.height=r.matSize[0],h().imshow(M,r),o.delete(),D.delete(),r.delete(),Q.delete(),E.delete()',
    to: guardedCleanupExpression,
  },
];

const backendCompatibilityPatches = [
  {
    from: 'A.isFloatTextureReadPixelsEnabledMethod=function(A,I,g){var C=A;',
    to: 'A.isFloatTextureReadPixelsEnabledMethod=function(A,I,g){try{var C=A;if(!C||"function"!=typeof C.framebufferTexture2D)return!1;',
  },
  {
    from: 'var i=C.checkFramebufferStatus(C.FRAMEBUFFER)===C.FRAMEBUFFER_COMPLETE;C.readPixels(0,0,1,1,C.RGBA,C.FLOAT,new Float32Array(4));var o=C.getError()===C.NO_ERROR;return i&&o}',
    to: 'var i=C.checkFramebufferStatus(C.FRAMEBUFFER)===C.FRAMEBUFFER_COMPLETE;C.readPixels(0,0,1,1,C.RGBA,C.FLOAT,new Float32Array(4));var o=C.getError()===C.NO_ERROR;return i&&o}catch(A){return!1}}',
  },
  {
    from: 'A.isDownloadFloatTextureEnabled=function(A,I){var g=A.createTexture();',
    to: 'A.isDownloadFloatTextureEnabled=function(A,I){try{if(!A||"function"!=typeof A.framebufferTexture2D)return!1;var g=A.createTexture();',
  },
  {
    from: 'var B=A.checkFramebufferStatus(A.FRAMEBUFFER)===A.FRAMEBUFFER_COMPLETE;return A.bindTexture(A.TEXTURE_2D,null),A.bindFramebuffer(A.FRAMEBUFFER,null),A.deleteTexture(g),A.deleteFramebuffer(C),B}',
    to: 'var B=A.checkFramebufferStatus(A.FRAMEBUFFER)===A.FRAMEBUFFER_COMPLETE;return A.bindTexture(A.TEXTURE_2D,null),A.bindFramebuffer(A.FRAMEBUFFER,null),A.deleteTexture(g),A.deleteFramebuffer(C),B}catch(A){return!1}}',
  },
];

let bundleSource = fs.readFileSync(ocrBundlePath, 'utf8');
let changed = false;

for (const repair of scalarRepairs) {
  if (!bundleSource.includes(repair.from)) {
    continue;
  }

  bundleSource = bundleSource.replace(repair.from, repair.to);
  changed = true;
}

for (const repair of cleanupRepairs) {
  if (!bundleSource.includes(repair.from)) {
    continue;
  }

  bundleSource = bundleSource.replace(repair.from, repair.to);
  changed = true;
}

for (const replacement of cleanupPatches) {
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

for (const replacement of backendCompatibilityPatches) {
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
  console.log('[patch-paddle-ocr-leaks] Applied OCR compatibility patch.');
} else {
  console.log('[patch-paddle-ocr-leaks] Patch already present.');
}
