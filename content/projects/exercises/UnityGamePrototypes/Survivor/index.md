+++
title = "Survivor"
summary = ""
description = ""
featuredImage = "survivors_03.png"
tags = ["Unity", "Survival"]
categories = ["exercises"]
collections = ["Unity Prototype"]
draft = false
+++

### Key features:
- The game offers two designed levels. At the beginning of the game, an additional welcome scene introduces players to the gameplay. In this scene, enemies are dynamically spawned and move around the map.
- There are two types of enemies that utilize AI Navigation, they can pursue the player intelligently while avoiding obstacles in their path.
- When an enemy approaches the player within a defined threshold, it transitions from pursuit to aggressive attacks. If the player manages to create distance beyond this threshold, the enemy reverts to chasing mode, adapting its movements to keep up with the player.
- Both the player and enemies possess a HealthSystem. A health bar is displayed to indicate the current health status of each entity. Successful hits result in health loss for them.
- This demo incorporates special effects, including particle systems and visual effects (VFX). These effects enhance key moments such as when the player or enemy gets hit. Additionally, atmospheric smoke effects is added.
- Customized Unity shaders are utilized to deliver the special effects in the game.
- The player and enemies feature multiple animations representing various behaviours and reactions. These animations provide visual cues during gameplay.

{{< button href="https://github.com/arialian/Survivors" target="_self" >}}
{{< icon "github" >}} View on Github
{{< /button >}}

###
{{< gallery >}}
  <img src="survivors_01.png" class="grid-w33" />
  <img src="survivors_02.png" class="grid-w33" />
  <img src="survivors_03.png" class="grid-w33" />
{{< /gallery >}}

