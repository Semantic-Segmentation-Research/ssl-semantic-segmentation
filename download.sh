#!/bin/bash

SERVER="root@ssh9.vast.ai"
LOCAL_SAVE="."
PORT="17536"

# 가져올 파일들의 절대 경로를 리스트로 쭉 나열하세요! (괄호 안에 작성)
FILES=("/home/dev/ssl-semantic-segmentation/ssl_main.py"
	"/home/dev/ssl-semantic-segmentation/model/resnet.py"
	"/home/dev/ssl-semantic-segmentation/semseg/context_module.py"
	"/home/dev/ssl-semantic-segmentation/semseg/deeplabv3plus.py"
	"/home/dev/ssl-semantic-segmentation/configuration.py"
)
# ==========================================

echo "📥 총 ${#FILES[@]}개의 파일을 다운로드합니다..."
	echo "----------------------------------------"

# 배열에 있는 파일을 하나씩 순회하면서 다운로드
for file in "${FILES[@]}"; do
	echo "👉 가져오는 중: $file"
	scp -P "${PORT}" "${SERVER}:${file}" "${LOCAL_SAVE}"

		# 다운로드 실패 시 알림
		if [ $? -ne 0 ]; then
			echo "⚠️ 경고: [ $file ] 다운로드 실패!"
		fi
	done

	echo "----------------------------------------"
	echo "✅ 모든 다운로드 작업이 완료되었습니다!"
