document.addEventListener('DOMContentLoaded', () => {
    const state = {
        results: [],
        selectedIndex: -1,
        selectedViewIndex: -1
    };

    const DOM = {
        loading: document.getElementById('loadingIndicator'),
        listContainer: document.getElementById('submittalListContainer'),
        viewerPlaceholder: document.getElementById('viewerPlaceholder'),
        viewerContent: document.getElementById('viewerContent'),
        viewerTitle: document.getElementById('viewerTitle'),
        badgePass: document.getElementById('badgePass'),
        badgeFail: document.getElementById('badgeFail'),
        badgeReview: document.getElementById('badgeReview'),
        gridBody: document.getElementById('gridBody'),
        submittalImg: document.getElementById('submittalImg'),
        archImg: document.getElementById('archImg'),
        archSelector: document.getElementById('archSelector'),
        
        // Modal
        btnNewAnalysis: document.getElementById('btnNewAnalysis'),
        uploadModal: document.getElementById('uploadModal'),
        btnCloseModal: document.getElementById('btnCloseModal'),
        uploadForm: document.getElementById('uploadForm'),
        uploadStatus: document.getElementById('uploadStatus'),
        btnSubmitRun: document.getElementById('btnSubmitRun')
    };

    // Modal Events
    DOM.btnNewAnalysis.addEventListener('click', () => DOM.uploadModal.classList.remove('hidden'));
    DOM.btnCloseModal.addEventListener('click', () => DOM.uploadModal.classList.add('hidden'));

    DOM.uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        DOM.btnSubmitRun.disabled = true;
        DOM.uploadStatus.classList.remove('hidden');

        const formData = new FormData(DOM.uploadForm);
        try {
            const response = await fetch('/api/run', {
                method: 'POST',
                body: formData
            });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.error || 'Execution failed');
            }
            const data = await response.json();
            state.results.unshift(data); // Add to top
            selectItem(0); // Select new item
            DOM.uploadModal.classList.add('hidden');
            DOM.uploadForm.reset();
        } catch (error) {
            alert("Error: " + error.message);
        } finally {
            DOM.btnSubmitRun.disabled = false;
            DOM.uploadStatus.classList.add('hidden');
        }
    });

    async function fetchResults() {
        try {
            const response = await fetch('/api/results');
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            state.results = data;
            DOM.loading.style.display = 'none';
            renderList();
        } catch (error) {
            console.error('Error fetching results:', error);
            DOM.loading.textContent = 'Error loading data';
            DOM.loading.style.color = 'var(--status-fail)';
        }
    }

    function renderList() {
        DOM.listContainer.innerHTML = '';
        if (state.results.length === 0) {
            DOM.listContainer.innerHTML = '<div style="padding: 16px; color: var(--text-muted);">No submittal results found.</div>';
            return;
        }

        state.results.forEach((item, index) => {
            const el = document.createElement('div');
            el.className = `submittal-item ${state.selectedIndex === index ? 'active' : ''}`;
            
            const filename = item.submittal_path.split(/[\\/]/).pop();
            const date = new Date(item.run_timestamp).toLocaleString();

            el.innerHTML = `
                <div class="submittal-item-title">${filename}</div>
                <div class="submittal-item-date">${date}</div>
                <div style="margin-top: 8px; display: flex; gap: 4px;">
                    <span style="font-size: 0.7rem; color: var(--status-pass)">${item.pass_count} P</span>
                    <span style="font-size: 0.7rem; color: var(--status-fail)">${item.fail_count} F</span>
                    <span style="font-size: 0.7rem; color: var(--status-review)">${item.review_required_count} R</span>
                </div>
            `;

            el.addEventListener('click', () => selectItem(index));
            DOM.listContainer.appendChild(el);
        });
    }

    function selectItem(index) {
        state.selectedIndex = index;
        state.selectedViewIndex = 0; // Default to first view's arch image
        renderList();
        renderViewer();
    }

    function selectViewForArchImage(viewIndex) {
        state.selectedViewIndex = viewIndex;
        // Highlight grid row
        const rows = DOM.gridBody.querySelectorAll('tr');
        rows.forEach((row, i) => {
            if (i === viewIndex) {
                row.style.background = 'rgba(255,255,255,0.1)';
            } else {
                row.style.background = 'transparent';
            }
        });
        
        // Update arch image
        const item = state.results[state.selectedIndex];
        const view = item.view_validation_results[viewIndex];
        if (view && view.arch_crop_path) {
            const urlPath = view.arch_crop_path.replace(/\\/g, '/');
            let afterInputs = urlPath;
            if (urlPath.includes('inputs/')) {
                afterInputs = urlPath.split('inputs/')[1];
            }
            DOM.archImg.src = `/api/images/inputs/${afterInputs}`;
        } else {
            DOM.archImg.src = '';
        }
    }

    function renderViewer() {
        const item = state.results[state.selectedIndex];
        if (!item) return;

        DOM.viewerPlaceholder.classList.add('hidden');
        DOM.viewerContent.classList.remove('hidden');

        DOM.viewerTitle.textContent = item.submittal_path;
        
        DOM.badgePass.textContent = `${item.pass_count} Pass`;
        DOM.badgeFail.textContent = `${item.fail_count} Fail`;
        DOM.badgeReview.textContent = `${item.review_required_count} Review`;

        // Render Grid
        DOM.gridBody.innerHTML = '';
        const views = item.view_validation_results || [];
        views.forEach((v, idx) => {
            const rule = v.rule_result;
            const status = rule ? rule.status : 'UNKNOWN';
            
            const tr = document.createElement('tr');
            tr.style.cursor = 'pointer';
            tr.innerHTML = `
                <td><strong>${v.view_id}</strong></td>
                <td>
                    <div style="font-weight: 500">${v.submittal_room_name || '-'}</div>
                    <div style="font-size: 0.8em; color: var(--text-muted)">${v.submittal_room_number || '-'}</div>
                </td>
                <td>
                    <div style="font-weight: 500">${(rule && rule.architectural_room_name) ? rule.architectural_room_name : '-'}</div>
                    <div style="font-size: 0.8em; color: var(--text-muted)">${(rule && rule.architectural_room_number) ? rule.architectural_room_number : '-'}</div>
                </td>
                <td class="status-cell status-${status}">${status}</td>
                <td class="reason-text">${rule ? rule.reason : 'N/A'}</td>
            `;
            tr.addEventListener('click', () => selectViewForArchImage(idx));
            DOM.gridBody.appendChild(tr);
        });

        // Set Submittal Image
        if (item.markup_image) {
            const urlPath = item.markup_image.replace(/\\/g, '/').replace('outputs/', '');
            DOM.submittalImg.src = `/api/images/outputs/${urlPath}`;
        } else {
            DOM.submittalImg.src = '';
        }

        // Collect unique arch images
        const uniqueArchPaths = new Set();
        views.forEach(v => {
            if (v.arch_crop_path) uniqueArchPaths.add(v.arch_crop_path);
        });

        // Render thumbnails
        DOM.archSelector.innerHTML = '';
        if (uniqueArchPaths.size > 0) {
            DOM.archSelector.classList.remove('hidden');
            let first = true;
            uniqueArchPaths.forEach(archPath => {
                const urlPath = archPath.replace(/\\/g, '/');
                let afterInputs = urlPath;
                if (urlPath.includes('inputs/')) afterInputs = urlPath.split('inputs/')[1];
                const imgSrc = `/api/images/inputs/${afterInputs}`;

                const img = document.createElement('img');
                img.src = imgSrc;
                img.className = 'arch-thumbnail';
                if (first) {
                    img.classList.add('active');
                    first = false;
                }
                img.addEventListener('click', () => {
                    // Update main arch image
                    DOM.archImg.src = imgSrc;
                    // Update active class
                    DOM.archSelector.querySelectorAll('img').forEach(i => i.classList.remove('active'));
                    img.classList.add('active');
                });
                DOM.archSelector.appendChild(img);
            });
        } else {
            DOM.archSelector.classList.add('hidden');
        }

        // Initialize arch image mapping from the first view
        if (views.length > 0) {
            selectViewForArchImage(0);
        } else {
            DOM.archImg.src = '';
        }
    }

    fetchResults();
});
