FROM sha256:7e60208fb45e8af3f1ab777cf24df7a7a415d8d84e2972ee76def4daf5c0f026
COPY umbrella/ /opt/saea-release/
COPY source-manifest.json /opt/saea-release/source-manifest.json
COPY verify-release.py /opt/saea-release/verify-release.py
COPY install-source.py /tmp/saea-install-source.py
ENV PYTHONPATH=/opt/saea-release/studies/speech-aware-evidence-acquisition/src PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 SAEA_RELEASE_MANIFEST=/opt/saea-release/source-manifest.json
WORKDIR /opt/saea-release/studies/speech-aware-evidence-acquisition
RUN /opt/saea-venv/bin/python /tmp/saea-install-source.py && python3 /tmp/saea-install-source.py && rm /tmp/saea-install-source.py && rm -rf /opt/umbrella/studies/speech-aware-evidence-acquisition && chmod -R a-w /opt/saea-release && /opt/saea-venv/bin/python /opt/saea-release/verify-release.py /opt/saea-release/source-manifest.json /opt/saea-release && /opt/saea-venv/bin/python -c 'from speech_aware_evidence_acquisition.core import scorers; assert scorers.__file__.startswith("/opt/saea-release/")' && python3 -c 'from speech_aware_evidence_acquisition.core import scorers; assert scorers.__file__.startswith("/opt/saea-release/")'
