{{- define "model-factory.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "model-factory.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "model-factory.name" . -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "model-factory.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "model-factory.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "model-factory.selectorLabels" -}}
app.kubernetes.io/name: {{ include "model-factory.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "model-factory.apiSelectorLabels" -}}
{{ include "model-factory.selectorLabels" . }}
app.kubernetes.io/component: api
{{- end -}}

{{- define "model-factory.workerSelectorLabels" -}}
{{ include "model-factory.selectorLabels" . }}
app.kubernetes.io/component: worker
{{- end -}}

{{- define "model-factory.secretName" -}}
{{- default (printf "%s-secrets" (include "model-factory.fullname" .)) .Values.secret.name -}}
{{- end -}}
