{{- define "tessera.name" -}}tessera{{- end -}}

{{- define "tessera.image" -}}
{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}
{{- end -}}

{{- define "tessera.labels" -}}
app.kubernetes.io/name: {{ include "tessera.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}

{{- define "tessera.secretEnv" -}}
{{- range .Values.keyVault.secrets }}
- name: {{ .env }}
  valueFrom:
    secretKeyRef:
      name: {{ $.Release.Name }}-kv
      key: {{ .objectName }}
{{- end }}
{{- end -}}

{{- define "tessera.volumes" -}}
- name: config
  configMap:
    name: {{ .Release.Name }}-config
- name: kv
  csi:
    driver: secrets-store.csi.k8s.io
    readOnly: true
    volumeAttributes:
      secretProviderClass: {{ .Release.Name }}-kv
- name: logs
  emptyDir: {}
- name: tmp
  emptyDir: {}
{{- end -}}

{{- define "tessera.mounts" -}}
- name: config
  mountPath: /etc/tessera
  readOnly: true
- name: kv
  mountPath: /mnt/secrets
  readOnly: true
- name: logs
  mountPath: /var/log/tessera
- name: tmp
  mountPath: /tmp
{{- end -}}
