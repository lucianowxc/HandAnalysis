const apiInput = document.getElementById('apiBaseUrl');
const saveApiBtn = document.getElementById('saveApiBtn');
const apiStatus = document.getElementById('apiStatus');
const csvForm = document.getElementById('csvForm');
const videoForm = document.getElementById('videoForm');
const resultEl = document.getElementById('result');
const linksEl = document.getElementById('links');
const previewEl = document.getElementById('preview');

const API_KEY = 'handanalysis_api_base_url';

function getApiBaseUrl() {
  return (localStorage.getItem(API_KEY) || '').replace(/\/$/, '');
}

function setApiBaseUrl(url) {
  localStorage.setItem(API_KEY, url.replace(/\/$/, ''));
}

function requireApiBaseUrl() {
  const base = getApiBaseUrl();
  if (!base) {
    throw new Error('Configure a API base URL antes de enviar.');
  }
  return base;
}

function clearResultViews() {
  linksEl.innerHTML = '';
  previewEl.innerHTML = '';
}

function renderResult(data) {
  resultEl.textContent = JSON.stringify(data, null, 2);
  clearResultViews();

  const files = data.files || {};
  Object.entries(files).forEach(([name, relUrl]) => {
    const a = document.createElement('a');
    a.href = `${getApiBaseUrl()}${relUrl}`;
    a.target = '_blank';
    a.rel = 'noopener';
    a.textContent = `${name}: ${a.href}`;
    linksEl.appendChild(a);
  });

  if (files.overlay) {
    const title = document.createElement('h3');
    title.textContent = 'Preview: vídeo com esqueleto';
    const video = document.createElement('video');
    video.controls = true;
    video.src = `${getApiBaseUrl()}${files.overlay}`;
    previewEl.appendChild(title);
    previewEl.appendChild(video);
  }

  if (files.motion) {
    const title = document.createElement('h3');
    title.textContent = 'Preview: gráfico de movimento';
    const img = document.createElement('img');
    img.alt = 'Motion plot';
    img.src = `${getApiBaseUrl()}${files.motion}`;
    previewEl.appendChild(title);
    previewEl.appendChild(img);
  }
}

async function submitForm(form, endpoint) {
  clearResultViews();
  resultEl.textContent = 'Processando...';

  const base = requireApiBaseUrl();
  const formData = new FormData(form);

  ['normalize'].forEach((name) => {
    const el = form.querySelector(`input[name="${name}"]`);
    if (el) formData.set(name, el.checked ? 'true' : 'false');
  });

  ['max_frames', 'model_asset_path'].forEach((name) => {
    const value = formData.get(name);
    if (value === '' || value === null) formData.delete(name);
  });

  const resp = await fetch(`${base}${endpoint}`, { method: 'POST', body: formData });
  const data = await resp.json();

  if (!resp.ok) {
    throw new Error(data?.detail || JSON.stringify(data));
  }

  renderResult(data);
}

saveApiBtn.addEventListener('click', () => {
  const url = apiInput.value.trim();
  if (!url) {
    apiStatus.textContent = 'Informe uma URL válida.';
    return;
  }
  setApiBaseUrl(url);
  apiStatus.textContent = `API URL salva: ${getApiBaseUrl()}`;
});

csvForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    await submitForm(csvForm, '/analyze/csv');
  } catch (err) {
    resultEl.textContent = `Erro: ${err.message || err}`;
  }
});

videoForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    await submitForm(videoForm, '/analyze/video');
  } catch (err) {
    resultEl.textContent = `Erro: ${err.message || err}`;
  }
});

apiInput.value = getApiBaseUrl();
if (apiInput.value) {
  apiStatus.textContent = `API URL carregada: ${apiInput.value}`;
}
