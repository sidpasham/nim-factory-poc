{{- define "llm-gpu-benchmarking.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "llm-gpu-benchmarking.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "llm-gpu-benchmarking.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "llm-gpu-benchmarking.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "llm-gpu-benchmarking.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "llm-gpu-benchmarking.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llm-gpu-benchmarking.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "llm-gpu-benchmarking.apiSelectorLabels" -}}
{{ include "llm-gpu-benchmarking.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end -}}

{{- define "llm-gpu-benchmarking.workerSelectorLabels" -}}
{{ include "llm-gpu-benchmarking.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end -}}

{{- define "llm-gpu-benchmarking.secretName" -}}
{{- default (printf "%s-secrets" (include "llm-gpu-benchmarking.fullname" .)) .Values.secret.name -}}
{{- end -}}
