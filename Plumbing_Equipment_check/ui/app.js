document.addEventListener('DOMContentLoaded', () => {
    // Nav Elements
    const navItems = document.querySelectorAll('.step-nav');
    const sections = document.querySelectorAll('.view-section');

    // Buttons
    const btnExtract = document.getElementById('btn-start-extract');
    const btnRunQC = document.getElementById('btn-run-qc');
    const btnReset = document.getElementById('btn-reset');

    // Terminal & Progress
    const termOutput = document.getElementById('term-output');
    
    // QC data will be loaded from the backend
    let qcReport = null;

    // Drag and Drop Logic
    const dropZones = document.querySelectorAll('.drop-zone');
    let uploadedCount = 0;

    dropZones.forEach(zone => {
        const input = zone.querySelector('.file-input');
        const fileNameEl = zone.querySelector('.file-name');

        // Click to upload
        zone.addEventListener('click', () => input.click());

        input.addEventListener('change', (e) => {
            if(e.target.files.length > 0) handleFile(zone, fileNameEl, e.target.files[0]);
        });

        // Drag and drop events
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if(e.dataTransfer.files.length > 0) handleFile(zone, fileNameEl, e.dataTransfer.files[0]);
        });
    });

    function handleFile(zone, nameEl, file) {
        if(file.type !== 'application/pdf') {
            alert('Please upload a PDF file.');
            return;
        }
        nameEl.textContent = file.name;
        if (!zone.classList.contains('ready')) {
            zone.classList.add('ready');
            uploadedCount++;
        }
        
        // Enable extract button if all 3 are uploaded
        if(uploadedCount >= 3) {
            btnExtract.disabled = false;
        }
    }

    // Navigation function
    function goToStep(step) {
        navItems.forEach(nav => nav.classList.remove('active'));
        sections.forEach(sec => sec.classList.remove('active'));
        
        document.querySelector(`.step-nav[data-step="${step}"]`).classList.add('active');
        document.getElementById(`step-${step}`).classList.add('active');
    }

    // Terminal Logging function
    function logTerm(msg, type = '') {
        const time = new Date().toLocaleTimeString('en-US', { hour12: false });
        const div = document.createElement('div');
        div.className = `log-line ${type}`;
        div.innerHTML = `<span class="time">[${time}]</span> <span class="script">${msg}</span>`;
        termOutput.appendChild(div);
        termOutput.scrollTop = termOutput.scrollHeight;
    }

    function typeLogSequence(logs, index, callback) {
        if(index >= logs.length) {
            if(callback) callback();
            return;
        }
        logTerm(logs[index].m, logs[index].t);
        setTimeout(() => typeLogSequence(logs, index+1, callback), logs[index].d || 400);
    }

    // Extraction simulation
    btnExtract.addEventListener('click', () => {
        goToStep(2);
        termOutput.innerHTML = '';
        
        // Reset progress
        for(let i=1; i<=3; i++) {
            document.getElementById(`prog-fill-${i}`).style.width = '0%';
            document.getElementById(`prog-val-${i}`).innerText = '0%';
        }

        const simLogs = [
            {m: "Initializing extraction engines...", t: "cyan", d: 800},
            {m: "[plumbing.py] Reading submittal_drawing.pdf...", t: "", d: 500},
            {m: "[plumbing16_extractor.py] Reading Plumbing16.pdf...", t: "", d: 400},
            {m: "[task3_extractor.py] Reading arc-100.pdf...", t: "", d: 900},
            {m: "[plumbing16_extractor.py] Vertical grid lines detected: [108, 804, 846, 938...]", t: "warn", d: 600},
            {m: "[plumbing16_extractor.py] Parsed columns: TAG, FIXTURE, SPEC, DESIGN...", t: "", d: 500},
            {m: "[task3_extractor.py] Found 10 unique annotations across 8 pages.", t: "", d: 700},
            {m: "[plumbing16_extractor.py] Extracted sub-components: FIXTURE, FAUCET, FLUSH VALVE.", t: "", d: 500},
            {m: "[plumbing.py] Extraction complete.", t: "success", d: 300},
            {m: "[plumbing16_extractor.py] Generated table extraction output/plumbing16/extracted_schedules.json", t: "success", d: 400},
            {m: "[task3_extractor.py] Generated ALL_PAGES_summary.json", t: "success", d: 800},
            {m: "All parallel extractions finished. Artifacts saved.", t: "cyan", d: 1000}
        ];

        // Animate Progress Bars
        setTimeout(() => { document.getElementById('prog-fill-1').style.width = '100%'; document.getElementById('prog-val-1').innerText = '100%'; }, 2500);
        setTimeout(() => { document.getElementById('prog-fill-2').style.width = '100%'; document.getElementById('prog-val-2').innerText = '100%'; }, 4200);
        setTimeout(() => { document.getElementById('prog-fill-3').style.width = '100%'; document.getElementById('prog-val-3').innerText = '100%'; }, 4900);

        typeLogSequence(simLogs, 0, () => {
            setTimeout(() => goToStep(3), 1500);
        });
    });

    // Run QC simulation -> Now calling actual Backend
    btnRunQC.addEventListener('click', async () => {
        btnRunQC.disabled = true;
        btnRunQC.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing with Qwen AI...';
        termOutput.innerHTML = '';
        logTerm("Executing qc_cross_check.py (this may take a few moments for Qwen to verify)...", "cyan");
        
        try {
            const res = await fetch('/run_qc', { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'success') {
                logTerm("QC Check Complete. Generating Report.", "success");
                qcReport = data.report;
                
                // Print python logs
                const logs = data.logs.split('\\n');
                logs.forEach(l => {
                    if(l.trim()) logTerm(l, l.includes('AI') ? 'warn' : '');
                });
                
                setTimeout(() => {
                    btnRunQC.disabled = false;
                    btnRunQC.innerHTML = 'Run qc_cross_check.py <i class="fa-solid fa-microchip"></i>';
                    renderDashboard();
                    goToStep(4);
                }, 1000);
            } else {
                logTerm("Error running QC: " + data.message, "danger");
                btnRunQC.disabled = false;
                btnRunQC.innerHTML = 'Retry QC';
            }
        } catch (e) {
            logTerm("Server Error: " + e.message, "danger");
            btnRunQC.disabled = false;
            btnRunQC.innerHTML = 'Retry QC';
        }
    });

    // Render Dashboard Lists
    function renderDashboard() {
        if (!qcReport) return;
        const mismatchList = document.getElementById('mismatch-list');
        const missingList = document.getElementById('missing-list');
        const matchedList = document.getElementById('matched-list');
        
        mismatchList.innerHTML = '';
        missingList.innerHTML = '';
        matchedList.innerHTML = '';

        // Update Stats
        const numMatched = qcReport.successfully_matched.length;
        const numMiss = qcReport.missing_from_master.length;
        const numMis = qcReport.information_mismatches.length;
        const total = numMatched + numMiss + numMis;
        
        const statCards = document.querySelectorAll('.stat-card h2');
        if(statCards.length >= 4) {
            statCards[0].textContent = total;
            statCards[1].textContent = numMatched;
            statCards[2].textContent = numMis;
            statCards[3].textContent = numMiss;
        }

        // 1. Render Mismatches
        qcReport.information_mismatches.forEach(item => {
            const group = document.createElement('div');
            group.className = 'issue-group';
            
            // Check if AI completely resolved this tag (all issues have "AI Resolved")
            let hasRealDiscrepancy = false;
            let issuesHtml = '';
            
            for(const [msg, pages] of Object.entries(item.issues)) {
                if(!msg.includes("AI Resolved")) {
                    hasRealDiscrepancy = true;
                }
                
                // Format the AI Notes nicely
                let formattedMsg = msg;
                if(msg.includes("(AI Note:")) {
                    const parts = msg.split("(AI Note:");
                    const aiText = parts[1].replace(")", "");
                    const isResolved = aiText.includes("Resolved");
                    const iconColor = isResolved ? "var(--accent-cyan)" : "var(--accent-red)";
                    formattedMsg = `${parts[0]} <br><span style='color: ${iconColor}; font-size: 0.85em; margin-top:0.5rem; display:block;'><i class='fa-solid fa-robot'></i> AI Note: ${aiText}</span>`;
                }
                
                issuesHtml += `
                    <div class="issue-item">
                        <div class="issue-msg">${formattedMsg}</div>
                        <div class="issue-pages">Found on pages: ${pages.join(', ')}</div>
                    </div>
                `;
            }

            const headerColor = hasRealDiscrepancy ? "yellow" : "green";

            group.innerHTML = `
                <div class="issue-header ${headerColor}" onclick="this.nextElementSibling.classList.toggle('open')">
                    <span class="tag">${item.tag}</span>
                    <span class="count">${Object.keys(item.issues).length} Issue(s) <i class="fa-solid fa-chevron-down" style="margin-left: 0.5rem"></i></span>
                </div>
                <div class="issue-body">
                    ${issuesHtml}
                </div>
            `;
            mismatchList.appendChild(group);
        });

        // 2. Render Missing
        qcReport.missing_from_master.forEach(item => {
            const group = document.createElement('div');
            group.className = 'issue-group';
            group.innerHTML = `
                <div class="issue-header red">
                    <span class="tag">${item.tag}</span>
                    <span class="count" style="color: var(--accent-red)"><i class="fa-solid fa-triangle-exclamation"></i> Missing</span>
                </div>
                <div class="issue-body open">
                    <div class="issue-item">
                        <div class="issue-msg">This tag appears on drawing floor plans but is missing from the master engineering schedules.</div>
                        <div class="issue-pages">Found on pages: ${item.pages.join(', ')}</div>
                    </div>
                </div>
            `;
            missingList.appendChild(group);
        });
        
        // 3. Render Successfully Matched
        qcReport.successfully_matched.forEach(tag => {
            const group = document.createElement('div');
            group.className = 'issue-group';
            group.innerHTML = `
                <div class="issue-header green">
                    <span class="tag">${tag}</span>
                    <span class="count" style="color: var(--accent-cyan)"><i class="fa-solid fa-check"></i> Verified</span>
                </div>
            `;
            matchedList.appendChild(group);
        });
    }

    btnReset.addEventListener('click', () => {
        goToStep(1);
    });
});
