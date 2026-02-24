unique_skill_checklist

1) unique_skill_checklist.csv / .json
   units.json に入っている武将ごとの「固有戦法ID/名前」を一覧化したものです。
   unique_skill_id の 'UNQ_' を外したものを unique_skill_name として出しています。

2) unique_skill_suspects.csv
   「固有戦法名が別の武将名と一致している」など、ズレの可能性が高い行だけ抽出したものです。
   例：ねね が 'UNQ_豊臣秀吉' になっている、など。

使い方（PowerShell）
  cd $env:USERPROFILE\Downloads
  # Excelで開くなら CSV
  start .\unique_skill_checklist\unique_skill_checklist.csv

次にやること
  - あなたが持っている正しい「固有戦法一覧（所持武将つき）」をこの形式の JSON に直して送ってくれれば、
    units.json と突き合わせて「units.json の unique_skill_id を自動で正しいものに修正」した版も作れます。
