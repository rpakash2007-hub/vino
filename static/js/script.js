/**
 * NephroScan AI Client-Side Application Script
 * Handles drag-and-drop, scan validation, live preview, and multi-stage diagnostic feedback.
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Auto-dismiss alerts
  const alertCloseBtns = document.querySelectorAll('.alert-close');
  alertCloseBtns.forEach(btn => {
    btn.addEventListener('click', function () {
      const alert = this.closest('.alert');
      if (alert) alert.style.display = 'none';
    });
  });

  // 2. Upload Zone Elements
  const dropzone = document.getElementById('uploadDropzone');
  const fileInput = document.getElementById('scanFileInput');
  const previewBox = document.getElementById('previewContainer');
  const previewImage = document.getElementById('previewImage');
  const previewFilename = document.getElementById('previewFilename');
  const previewFilesize = document.getElementById('previewFilesize');
  const previewDimensions = document.getElementById('previewDimensions');
  const removeBtn = document.getElementById('removeScanBtn');
  const uploadForm = document.getElementById('scanUploadForm');
  const analyzeBtn = document.getElementById('analyzeScanBtn');
  const processingOverlay = document.getElementById('processingOverlay');
  const validationError = document.getElementById('validationError');

  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/jpg'];
  const MAX_BYTES = 16 * 1024 * 1024; // 16 MB

  if (dropzone && fileInput) {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, preventDefaults, false);
      document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    // Highlight drop area
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', function (e) {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0) {
        handleFileSelection(files[0]);
      }
    });

    // Handle file input change
    fileInput.addEventListener('change', function () {
      if (this.files && this.files.length > 0) {
        handleFileSelection(this.files[0]);
      }
    });

    // Handle remove
    if (removeBtn) {
      removeBtn.addEventListener('click', function () {
        fileInput.value = '';
        if (previewBox) previewBox.style.display = 'none';
        if (dropzone) dropzone.style.display = 'block';
        if (analyzeBtn) analyzeBtn.disabled = true;
        if (validationError) validationError.style.display = 'none';
      });
    }

    function handleFileSelection(file) {
      if (validationError) validationError.style.display = 'none';

      // Type check
      if (!ALLOWED_TYPES.includes(file.type)) {
        showError("Invalid format. Please select a valid kidney CT or MRI image (.jpg, .jpeg, or .png).");
        fileInput.value = '';
        return;
      }

      // Size check
      if (file.size > MAX_BYTES) {
        showError(`File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). Maximum allowed size is 16 MB.`);
        fileInput.value = '';
        return;
      }

      // Read & Preview
      const reader = new FileReader();
      reader.onload = function (e) {
        const img = new Image();
        img.onload = function () {
          // Dimension check
          if (this.width < 32 || this.height < 32) {
            showError("Image resolution is too low (< 32x32) for medical scan feature extraction.");
            fileInput.value = '';
            return;
          }

          if (previewImage) previewImage.src = e.target.result;
          if (previewFilename) previewFilename.textContent = file.name;
          if (previewFilesize) previewFilesize.textContent = formatBytes(file.size);
          if (previewDimensions) previewDimensions.textContent = `${this.width} × ${this.height} px`;

          if (dropzone) dropzone.style.display = 'none';
          if (previewBox) previewBox.style.display = 'block';
          if (analyzeBtn) analyzeBtn.disabled = false;
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }

    function showError(msg) {
      if (validationError) {
        validationError.textContent = msg;
        validationError.style.display = 'block';
      } else {
        alert(msg);
      }
    }

    function formatBytes(bytes) {
      if (bytes < 1024) return bytes + ' Bytes';
      else if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
      else return (bytes / 1048576).toFixed(2) + ' MB';
    }
  }

  // 3. Multi-Stage Diagnostic Progress Overlay
  if (uploadForm && analyzeBtn && processingOverlay) {
    uploadForm.addEventListener('submit', function (e) {
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        e.preventDefault();
        alert("Please select a kidney scan before analyzing.");
        return;
      }

      // Show overlay
      processingOverlay.style.display = 'flex';

      const steps = [
        { id: 'stage1', text: 'Analyzing Kidney Image...' },
        { id: 'stage2', text: 'Preprocessing image & normalizing...' },
        { id: 'stage3', text: 'Extracting reproducible radiodensity features...' },
        { id: 'stage4', text: 'Running Support Vector Machine (SVM)...' },
        { id: 'stage5', text: 'Running Decision Tree Classifier...' },
        { id: 'stage6', text: 'Ensembling predictions & generating report...' }
      ];

      let currentStep = 0;
      const interval = setInterval(() => {
        currentStep++;
        if (currentStep < steps.length) {
          const prevEl = document.getElementById(steps[currentStep - 1].id);
          const currEl = document.getElementById(steps[currentStep].id);
          if (prevEl) {
            prevEl.classList.remove('active');
            prevEl.classList.add('completed');
          }
          if (currEl) {
            currEl.classList.add('active');
          }
        } else {
          clearInterval(interval);
        }
      }, 550);
    });
  }

  // 4. History Table Search & Filter
  const historySearch = document.getElementById('historySearchInput');
  if (historySearch) {
    historySearch.addEventListener('keyup', function () {
      const filter = this.value.toLowerCase();
      const rows = document.querySelectorAll('.history-table-row');
      rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(filter) ? '' : 'none';
      });
    });
  }
});
