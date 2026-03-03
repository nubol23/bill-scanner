const { chromium } = require('playwright');

const APP_URL = process.env.APP_URL || 'http://127.0.0.1:4173/';
const MANUAL_SERIAL = '12345678';

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    await page.goto(APP_URL, { waitUntil: 'networkidle' });

    await page.getByRole('button', { name: 'Entrada Manual' }).click();
    await page.getByLabel('Número de serie manual').fill(MANUAL_SERIAL);
    await page.getByRole('button', { name: 'Revisar' }).click();

    await page.waitForFunction(
      (expectedSerial) => document.body.innerText.includes(expectedSerial),
      MANUAL_SERIAL,
      { timeout: 10_000 },
    );

    const bodyText = await page.locator('body').innerText();
    if (!bodyText.includes(`Billete de Bs 10`)) {
      throw new Error('The manual result card did not render.');
    }

    if (!bodyText.includes(MANUAL_SERIAL)) {
      throw new Error(`Expected ${MANUAL_SERIAL} in the UI.`);
    }
  } finally {
    await browser.close();
  }
})();
