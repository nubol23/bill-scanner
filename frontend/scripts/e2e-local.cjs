const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const APP_URL = process.env.APP_URL || 'http://127.0.0.1:4173/';
const IMAGES_DIR = path.resolve(__dirname, '..', '..', 'imgs');

function extractExpectedSerial(fileName) {
  const match = fileName.match(/(\d{8,9})/);
  return match ? match[1] : null;
}

(async () => {
  if (!fs.existsSync(IMAGES_DIR)) {
    console.log('Skipping E2E: imgs/ is not available.');
    process.exit(0);
  }

  const files = fs
    .readdirSync(IMAGES_DIR)
    .filter((file) => file.endsWith('.png'))
    .sort();

  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    await page.goto(APP_URL, { waitUntil: 'networkidle' });

    for (const file of files) {
      const expected = extractExpectedSerial(file);
      if (!expected) {
        continue;
      }

      const fileInput = page.locator('input[type="file"]').first();
      await fileInput.setInputFiles(path.join(IMAGES_DIR, file));
      await page.waitForFunction(
        () => !document.body.textContent.includes('Consultando OCR remoto...'),
        undefined,
        { timeout: 30000 },
      );

      const bodyText = await page.locator('body').innerText();
      if (bodyText.includes('CORS') || bodyText.includes('cross-origin')) {
        throw new Error(`CORS failure surfaced in UI while testing ${file}.`);
      }
      if (bodyText.includes('OCR tardó demasiado')) {
        throw new Error(`Legacy browser OCR timeout still surfaced in UI while testing ${file}.`);
      }
      if (!bodyText.includes(expected)) {
        throw new Error(`Expected ${expected} in UI for ${file}, but it was not found.`);
      }
    }
  } finally {
    await browser.close();
  }
})();
