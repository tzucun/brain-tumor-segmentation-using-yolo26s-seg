const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const artifacts = path.join(__dirname, '..', 'artifacts', 'qa');
  await fs.mkdir(artifacts, { recursive: true });
  const browser = await chromium.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', headless: true,
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.setDefaultTimeout(90000);
  async function settle() {
    await page.evaluate(() => new Promise(resolve => {
      let timer;
      const done = () => { observer.disconnect(); resolve(); };
      const observer = new MutationObserver(() => { clearTimeout(timer); timer = setTimeout(done, 700); });
      observer.observe(document.querySelector('.stApp'), { childList: true, subtree: true, attributes: true });
      timer = setTimeout(done, 700);
    }));
  }
  async function checkImages() {
    await page.waitForFunction(() => {
      const images = [...document.querySelectorAll('.prediction-image img')];
      return images.length === 3 && images.every(image => image.complete && image.naturalWidth > 0);
    });
    return page.locator('.prediction-image img').evaluateAll(elements => elements.map(image => {
      const canvas = document.createElement('canvas');
      canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(image, 0, 0);
      const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
      let visible = 0;
      for (let i = 0; i < pixels.length; i += 4) if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 60) visible++;
      return { width: image.naturalWidth, height: image.naturalHeight, visible };
    }));
  }
  async function checkTextColors(dark) {
    const colors = await page.locator('.stButton button:not(:disabled) p,.stDownloadButton button p,[data-testid="stFileUploaderDropzone"] button p')
      .evaluateAll(elements => elements.map(e => ({ text: e.textContent, color: getComputedStyle(e).color })));
    assert(colors.every(e => e.color === 'rgb(255, 255, 255)'), JSON.stringify(colors));
    const nameColor = await page.locator('[data-testid="stFileUploaderFileName"]').evaluate(e => getComputedStyle(e).color);
    assert.equal(nameColor, dark ? 'rgb(248, 250, 252)' : 'rgb(31, 41, 55)');
    const tableColors = await page.locator('.theme-table td').evaluateAll(elements => elements.map(e => getComputedStyle(e).color));
    assert(tableColors.every(color => color === (dark ? 'rgb(248, 250, 252)' : 'rgb(31, 41, 55)')));
  }
  try {
    await page.goto(process.env.APP_URL || 'http://127.0.0.1:8501', { waitUntil: 'domcontentloaded' });
    await page.getByRole('heading', { name: /Segmentasi Tumor Otak pada Citra MRI Dataset/ }).waitFor();
    await settle();
    assert.equal(await page.getByRole('tab').count(), 0);
    assert.equal(await page.getByRole('slider').count(), 1);
    assert.equal(await page.getByRole('combobox').count(), 0);
    assert.equal(await page.getByRole('radio').count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Jalankan prediksi', exact: true }).isDisabled(), true);
    assert.equal(await page.locator('[data-testid="stExpander"] details').getAttribute('open'), null);
    await page.locator('input[type="file"]').setInputFiles(path.join(__dirname, '..', 'samples', 'test_images', 'meningioma_sample.jpg'));
    await settle();
    assert.equal(await page.locator('.prediction-image img').count(), 0);
    await page.getByRole('button', { name: 'Jalankan prediksi', exact: true }).click();
    await page.getByRole('heading', { name: 'Hasil prediksi', exact: true }).waitFor();
    await page.locator('.stDownloadButton button').last().waitFor({ state: 'attached' });
    const images = await checkImages();
    assert(images.every(image => image.visible > 1000), JSON.stringify(images));
    const positions = await page.locator('.prediction-image').evaluateAll(elements => elements.map(e => {
      const b = e.getBoundingClientRect(); return { x: b.x, y: b.y };
    }));
    assert(positions[0].x < positions[1].x && positions[1].x < positions[2].x);
    assert(Math.abs(positions[0].y - positions[2].y) < 3);
    const headings = await page.locator('[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong').allTextContents();
    assert.deepEqual(headings.slice(0, 3), ['Citra masukan', 'Dengan negative samples', 'Tanpa negative samples']);
    const details = page.locator('[data-testid="stExpander"]').filter({
      has: page.getByText('Detail Hasil Prediksi dan Visualisasi', { exact: true }),
    });
    assert.equal(await details.locator('details').first().getAttribute('open'), null);
    assert.equal(await page.getByRole('heading', { name: 'Visualisasi hasil prediksi', exact: true }).isVisible(), false);
    await page.screenshot({ path: path.join(artifacts, 'results-collapsed-light.png'), fullPage: true });
    await details.locator('summary').first().click();
    await page.waitForFunction(() => document.querySelectorAll('[data-testid="stVegaLiteChart"] canvas,[data-testid="stVegaLiteChart"] svg').length >= 2);
    const chartMarks = await page.locator('[data-testid="stVegaLiteChart"]').evaluateAll(elements => elements.map(element => {
      const canvas = element.querySelector('canvas');
      if (!canvas) return element.querySelectorAll('svg .mark-rect path').length;
      const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
      let colored = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 3] && Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) - Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) > 35) colored++;
      }
      return colored;
    }));
    assert(chartMarks.length === 2 && chartMarks.every(count => count > 0), JSON.stringify(chartMarks));
    const thumbnail = page.locator('.prediction-image').nth(1);
    await thumbnail.hover();
    await page.waitForFunction(() => {
      const matrix = getComputedStyle(document.querySelectorAll('.prediction-image img')[1]).transform;
      return matrix !== 'none' && new DOMMatrix(matrix).a > 1.03;
    });
    const imageBounds = await thumbnail.boundingBox();
    await thumbnail.click();
    const enlarged = page.locator('.image-lightbox:popover-open');
    await enlarged.waitFor();
    assert((await enlarged.boundingBox()).width > imageBounds.width * 2);
    await page.screenshot({ path: path.join(artifacts, 'image-enlarged-light.png') });
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.image-lightbox:popover-open').count(), 0);
    await thumbnail.click();
    await page.getByRole('button', { name: 'Tutup gambar', exact: true }).click();
    assert.equal(await page.locator('.image-lightbox:popover-open').count(), 0);
    await checkTextColors(false);
    await page.screenshot({ path: path.join(artifacts, 'simple-light.png'), fullPage: true });
    for (const [index, suffix] of ['dengan_negative.png', 'tanpa_negative.png'].entries()) {
      const waiting = page.waitForEvent('download');
      await page.getByRole('button', { name: 'Unduh gambar hasil prediksi', exact: true }).nth(index).click();
      const download = await waiting;
      assert(download.suggestedFilename().endsWith(suffix));
      const filename = path.join(artifacts, download.suggestedFilename());
      await download.saveAs(filename);
      const png = await fs.readFile(filename);
      assert(png.length > 1000 && png.subarray(1, 4).toString() === 'PNG');
    }
    await page.getByText('Mode gelap', { exact: true }).click();
    await page.waitForFunction(() => getComputedStyle(document.querySelector('.stApp')).backgroundColor === 'rgb(15, 23, 42)');
    await settle();
    await checkImages();
    await checkTextColors(true);
    await page.screenshot({ path: path.join(artifacts, 'simple-dark.png'), fullPage: true });
    await page.getByText('Kinerja Model pada Dataset Uji', { exact: true }).click();
    await settle();
    assert.equal(await page.locator('[data-testid="stMetricValue"]').count(), 8);
    await checkTextColors(true);
    await page.getByText('Kinerja Model pada Dataset Uji', { exact: true }).click();
    if (await details.locator('details').first().getAttribute('open') !== null) {
      await details.locator('summary').first().click();
    }
    assert.equal(await details.locator('details').first().getAttribute('open'), null);
    await page.setViewportSize({ width: 390, height: 844 });
    await settle();
    await checkImages();
    const mobile = await page.locator('.prediction-image').evaluateAll(elements => elements.map(e => {
      const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, width: b.width, height: b.height };
    }));
    assert(mobile[0].y + mobile[0].height <= mobile[1].y && mobile[1].y + mobile[1].height <= mobile[2].y);
    assert(mobile.every(b => b.x >= 0 && b.x + b.width <= 392));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2), false);
    await page.screenshot({ path: path.join(artifacts, 'simple-mobile-dark.png'), fullPage: true });
    await page.getByRole('heading', { name: 'Hasil prediksi', exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(artifacts, 'simple-mobile-results.png') });
    await page.locator('.prediction-image').last().click();
    await page.locator('.image-lightbox:popover-open').waitFor();
    const modalBounds = await page.locator('.image-lightbox:popover-open').boundingBox();
    assert(modalBounds.x >= 0 && modalBounds.x + modalBounds.width <= 392);
    await page.screenshot({ path: path.join(artifacts, 'image-enlarged-mobile.png') });
    await page.getByRole('button', { name: 'Tutup gambar', exact: true }).click();
    const body = await page.locator('[data-testid="stMain"]').innerText();
    assert(!/bandingkan|membandingkan|perbandingan|Eksplorasi threshold|Citra contoh/.test(body));
    assert.equal(await page.locator('[data-testid="stException"]').count(), 0);
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ status: 'passed', images, chartMarks, errors, artifacts }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
