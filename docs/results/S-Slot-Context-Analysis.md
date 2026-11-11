# 槽位值、多类型与上下文关系分析

## 判断

同一个槽位值对应多个类型，本身是合理且必要的：类型标注的是表达在当前请求中的语义角色，不只是字面词义。但是现有多类型关系混合了真实多义、业务角色变化、类别定义交叉以及标注错误，不能全部解释为模型需要学会的上下文能力。

更合适的任务是P(type存在 | 完整语音query, intent上下文, type定义与区分规则)，而不是P(type | 槽值词面)或“给定intent即可确定所有槽位”。意图能大幅缩小角色假设，但未规定这个样本实际出现了哪些角色；不应把训练中未见的组合硬设为不可能。

## 全训练集与全测试集统计

按独立文本、原entity span的值统计，大小写/NFKC/空白归一，保留标点。先在clean-v6审查，再按用户持续清理明确错误的授权生成clean-v7；以下为更新后的完整数据，不是抽样。原始及v6统计保存在前一报告，不覆盖。

| 项目 | TRAIN | TEST | 两集合并去重 |
|---|---:|---:|---:|
| 有效文本数 | 11,460 | 2,947 | 14,407 |
| 槽位实体出现次数 | 11,324 | 2,799 | 14,123 |
| 相同值对应多个类型：不同值数量 | **170** | **25** | **197** |
| 仅在不同意图之间出现类型差异 | 137 | 20 | 152 |
| 至少一个相同意图内也有多类型 | **33** | **5** | **45** |
| 多类型值涉及实体出现次数 | 1,617 | 103 | 2,137 |
| 多类型值涉及文本数 | 1,532 | 101 | 2,023 |

合并列重新判断跨集合关系，不是前两列相加。有些值在各集合内单一类型，跨集合才出现不同类型。170/25是不同槽值数，不是错误样本数；33/5是其中子集。

TRAIN中137/170=80.6%的多类型值，仅在不同意图间观察到类型变化；TEST为20/25=80.0%。这支持意图与类型关联显著，但“同一意图内只看到一种”并不证明唯一映射：很多条件组样本少。

对多类型值的实体出现计算描述性条件熵：TRAIN H(type|value)=0.485 bits，H(type|value,intent)=0.068 bits；TEST为0.718→0.134。不能称为模型准确率提高86%或81%，也不能当因果证据。TRAIN的495个value-intent组中285组仅有一次观察；TEST60组中41组只有一次，切分本身会机械降低经验熵。当前没有用独立验证实验量化完整上下文贡献，因此不报告“多少错误可被上下文解决”的伪精确比例。

## 逐句区分四种机制

### 1. 词义或业务角色真正不同：应保留

**同意图也能合理改变类型。** TRAIN7439、8893均为calendar_set：

- “meeting with anna this friday”：anna=person，with引入参与者。
- “add new event to calendar and name as anna”：anna=event_name，name as明确命名事件。

TRAIN11436、5254均为play_music：

- “play kids songs”：kids=music_genre，描述歌曲类别/受众；其具体genre定义仍需依数据规范解释。
- “start playing kids by mgmt”：kids=song_name，by MGMT及播放对象构成作品指代。

TRAIN7406、8074均为calendar_set：

- “set repeating event gym day on calendar”：gym被用作活动内容。
- “meet sarah at the gym”：gym=place_name，at引入地点。

跨意图更明显：pink在灯光控制中是color_type，在音乐播放中是artist_name，在“does pink have a new baby”中标person。这不是通过统一词典类型能解决的问题。

**处理：** 保留这些训练对照，知识卡说明“什么关系使该类型成立”，few-shot同时展示同词不同角色；不能因多类型冲突删掉合理多义。

### 2. 本体层级与业务角色交叉：不能用上下文强行解释全部差异

news_query中的Trump既是person，又可作为news_topic；Syria/France既是place_name，又可成为新闻主题。人物/地点是实体本体类别，news_topic是请求中的话题角色，它们本来可能同时成立。数据却通常只给一个类型，必须确认标注优先级。

例如TRAIN3747“news about trump”标news_topic，4883“whats going on with trump”标person。上下文虽然不同，但均可能在询问这个人的新闻，不能笼统声称前者话题、后者人物就已解释一致性。

food_type/ingredient同样如此：TRAIN10071“cook a whole chicken”标food_type，9985“make steak with chicken”标ingredient，存在合理成品/配料差别；但“how to cook rice”与“explain the rice recipe”的类型不同，仅凭当前文本并没有足够清晰的区分。

**处理：** 卡片应区分本体属性、句内角色和数据集输出类别，明确输出优先级。若原始定义不明确，保留review_pending；不可为了提高exact随意选一边，也不可直接全改多标签，后者会变更评测任务。

### 3. 同义任务表达却标注不一致：上下文不足以解释

- TRAIN2904“please clean the floor”标place_name；TRAIN226“clean the floor please”标house_place。语序变化没有改变floor的角色。
- TRAIN11884“what's happening tonight in chicago”标timeofday；11890“what's happening in town tonight”标date。地点具体程度不能解释今晚类型变化。
- TRAIN4425“weekly weather”标date；4500“weekly weather report”标general_frequency。需要统一周期范围与重复执行的边界。
- TRAIN11657/11850都询问Logan的show times，分别movie_name/event_name。这至少是强冲突线索，不能归因缺少模型思考。

**处理：** 审查类别边界和完整标注后建立版本化规则。以上不是“已确认所有记录错误”的名单；在确定哪侧违反定义前不自动删整个词的样本。

### 4. 明确类型错误：已排除

本轮相对clean-v6新增：

| 集合/ID | 原始错误 | 上下文为何不能解释 | 录音数 |
|---|---|---|---:|
| TRAIN2228 | cst/est→place_name | convert nine am cst to est time明确时区转换 | 1 |
| TRAIN8595 | tuesday→event_name | on tuesday morning明确日期修饰 | 2 |
| TEST17119 | amazon→person | 邮件公司发送方不因此变成人名 | 1 |
| TEST16178 | amazon→person | 同上，完整句没有人名证据 | 1 |
| TEST1832 | coldplay→playlist_name | play coldplay album没有歌单指代 | 1 |

此轮新增TRAIN2句/3录音，TEST3句/3录音。clean-v7累计TRAIN删除54句/240录音，TEST删除27句/87录音；保留TRAIN11,460句/50,388录音，TEST2,947句/12,991录音。对应相同合成标注累计删除59句/151录音。原始数据保留，删除作用于有效样本视图；CURRENT、共享盘、配置和公开清单已同步。未音频听审，不宣称音频损坏。

## 对S节点的具体调整建议

1. **保留意图作为语义上下文，不把它当封闭类型白名单。** 是否先跑P属于架构和算力选择；相关性不等于必须串行依赖。
2. **知识卡从“类型词表”改成“角色判据”。** 每张卡包含语义定义、句内成立证据、与相邻类型的区别、合理缺省和冲突处理。例如person与event_name用参与者/命名对象关系区分。
3. **召回相似问时保存完整query及span角色。** 同词两条角色不同的例子，可能比很多同类型同词重复例更有价值。训练来源审查通过才能进入知识卡；测试仅用于质量审计，不进入教师材料或训练示例。
4. **独立二分类要保留完整query和必要的区分规则。** 可隐藏其他候选的预测，不能把与当前type判别有关的角色对照全部mask掉。否则独立判别很容易同时把人物和话题都判真，违反现有单span标注优先级。
5. **输出仍只包含类型存在与否。** 模型无需输出槽值，但内部判别不能忽略词语之间的语义关系；D负责最后的意图、槽值及整体一致性。
6. **将评测分层。** 除类型集合exact/F1，分别观察普通样本、真实多义、同意图角色对照、待审标注边界；待审状态不等于自动从评测移除，只有确认错误才进入删除清单。

优先级：先修明确的标注/知识冲突，再把合理多义组织为对照知识与few-shot，之后才以固定数据比较独立判别和联合判别。不能用当前混合噪声直接证明“每个槽位完全独立”或“P与S不相关”。本轮没有新NPU或教师实验。
