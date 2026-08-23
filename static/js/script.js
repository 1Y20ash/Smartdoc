const fileInput = document.getElementById('document');
const uploadZone = document.querySelector('.upload-zone');
const filename = document.getElementById('filename');
const dropTitle = document.getElementById('dropTitle');
const form = document.getElementById('uploadForm');
const analyzeButton = document.getElementById('analyzeButton');
const installPrompt = document.getElementById('installPrompt');
const installButton = document.getElementById('installButton');
const installClose = document.getElementById('installClose');
let deferredInstallPrompt = null;

function showFile(file) {
  if (!file) return;
  filename.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  dropTitle.textContent = 'Document ready to analyze';
  uploadZone.classList.add('file-selected');
}

fileInput?.addEventListener('change', () => showFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach(eventName => uploadZone?.addEventListener(eventName, event => { event.preventDefault(); uploadZone.classList.add('is-dragging'); }));
['dragleave', 'drop'].forEach(eventName => uploadZone?.addEventListener(eventName, event => { event.preventDefault(); uploadZone.classList.remove('is-dragging'); }));
uploadZone?.addEventListener('drop', event => {
  const file = event.dataTransfer.files[0];
  if (!file || !/\.(pdf|docx|txt)$/i.test(file.name)) return;
  const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files; showFile(file);
});
form?.addEventListener('submit', () => { analyzeButton.disabled = true; analyzeButton.innerHTML = 'Analyzing document <b>…</b>'; });

// Show the install prompt once per browser session, but never when SmartDoc is already installed.
function isStandalone() {
  return window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;
}
function hideInstallPrompt() {
  if (!installPrompt) return;
  installPrompt.classList.remove('show');
  installPrompt.setAttribute('aria-hidden', 'true');
}
function showInstallPrompt() {
  if (!installPrompt || isStandalone() || sessionStorage.getItem('smartdocInstallPromptShown') === '1') return;
  installPrompt.classList.add('show');
  installPrompt.setAttribute('aria-hidden', 'false');
  sessionStorage.setItem('smartdocInstallPromptShown', '1');
}

window.addEventListener('beforeinstallprompt', event => {
  event.preventDefault();
  deferredInstallPrompt = event;
  showInstallPrompt();
});

installButton?.addEventListener('click', async () => {
  if (!deferredInstallPrompt) {
    hideInstallPrompt();
    return;
  }
  deferredInstallPrompt.prompt();
  await deferredInstallPrompt.userChoice;
  deferredInstallPrompt = null;
  hideInstallPrompt();
});
installClose?.addEventListener('click', hideInstallPrompt);
window.addEventListener('appinstalled', () => { deferredInstallPrompt = null; hideInstallPrompt(); });

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/static/service-worker.js', { scope: '/' }).catch(error => console.warn('SmartDoc service worker registration failed:', error)));
}
