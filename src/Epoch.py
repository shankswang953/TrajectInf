from pickletools import optimize
from syslog import LOG_ERR
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import gc
from Args import get_args
from src.DataLoad import create_mixture_gaussian
from src.TrainLoss import short_term_loss, long_term_loss, concatenate_long_term_loss

def train_epoch(func, primary_data, time_points, time_steps_list, density_precompute_list, 
               optimizer, device, total_primal, manifold_mixture, args, secondary_data=None, secondary_mixture=None, writer=None, loss_meter=None, scheduler=None, epoch=0, rare_files=None):
    
    try:
        func.train()
        
        total_loss = 0
        short_term_total_loss = 0
        long_term_total_loss = 0
        
        
        progress_bar = tqdm(range(1, args.niters + 1), desc=f"Epoch {epoch}")
        
        for itr in progress_bar:

            optimizer.zero_grad()
            short_term_total_loss = 0
            all_section_forward_density_losses = []
            section_short_term_losses = []
            # short term loss
            for t_idx in range(len(time_points) - 1):
                short_results = short_term_loss(
                    func=func,
                    t_idx=t_idx,
                    time_points=time_points,
                    primary_data=primary_data,
                    secondary_data=secondary_data,
                    manifold_mixture=manifold_mixture,
                    secondary_mixture=secondary_mixture,
                    time_steps_list=time_steps_list,
                    density_precompute_list=density_precompute_list,
                    device=device,
                    args=args, 
                    writer=writer,
                    itr=itr + epoch * args.niters,
                    rare_files=rare_files
                )
                
                # Accumulate short term loss
                current_short_term_loss = short_results['total_loss']
                
                # Collect terminal density loss (for evaluation)
                if 'forward_losses' in short_results and 'terminal_density' in short_results['forward_losses']:
                    all_section_forward_density_losses.append(
                        short_results['forward_losses']['terminal_density'].item()
                    )
                    
                short_term_total_loss += current_short_term_loss
                section_short_term_losses.append(current_short_term_loss.item())
                
                del short_results
                

            
            # long term loss
            if args.sync_loss:
                total_secondary = torch.cat(secondary_data, dim=0)
            else:
                total_secondary = None
                  
                    
            if args.concatenate_loss:
                concatenate_loss = concatenate_long_term_loss(
                    func, 
                    primary_data,
                    secondary_data,
                    time_points,
                    time_steps_list,
                    density_precompute_list,
                    total_primal,  
                    total_secondary,
                    secondary_mixture,
                    args.support_points,
                    device,
                    args,  
                    writer=writer,
                    itr=itr + epoch * args.niters
                ) 
                
                long_term_loss = concatenate_loss * args.concatenate_coefficient
                short_term_total_loss += long_term_loss
            
            short_term_total_loss.backward()
            
            optimizer.step()
            
            
            # Update progress bar description
            progress_bar.set_postfix(
                short=f"{sum(section_short_term_losses)/(len(time_points)-1):.4f}",
            )
            if args.concatenate_loss:
                progress_bar.set_postfix(
                    long=f"{long_term_loss.item():.4f}"
                )
            
            # Record total loss
            if writer is not None:
                writer.add_scalar('Loss/total', sum(section_short_term_losses), itr + epoch * args.niters)
                if args.concatenate_loss:
                    writer.add_scalar('Loss/long', long_term_loss.item(), itr + epoch * args.niters)
                

            
            if itr < args.niters/2:
                scheduler.step()

            torch.cuda.empty_cache()
            gc.collect()
            loss_meter.update(total_loss)
            
            if args.train_dir is not None:
                # Save checkpoint every 1000 iterations
                if itr % 100 == 0:
                    if not os.path.exists(args.train_dir):
                        os.makedirs(args.train_dir, exist_ok=True)
                    # Save with iteration number in filename
                    ckpt_path = os.path.join(args.train_dir, f'ckpt_iter_{itr}.pth')
                    torch.save({
                        'func_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'iteration': itr,
                        'loss': loss_meter.avg
                    }, ckpt_path)
                    print('Stored checkpoint at {}'.format(ckpt_path))
                    # Calculate average terminal density loss
                    
            short_term_total_loss = sum(section_short_term_losses)
            avg_section_density_loss = short_term_total_loss / len(all_section_forward_density_losses) 
            if itr % 5:
                print(f"Avg terminal density loss: {avg_section_density_loss:.4f}")
                
                
            if itr % 5 == 0:
                print(f"Iter {itr}: short_term_loss={short_term_total_loss/(len(time_points)-1):.4f}")
                print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Iter {itr}: Current Learning Rate: {current_lr}')
                print('----------------------------------')
            
                        
                        
        
        
                
    except KeyboardInterrupt:

        os.makedirs(args.train_dir, exist_ok=True)
        ckpt_path = os.path.join(args.train_dir, 'ckptBreak.pth')
        torch.save({
            'func_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, ckpt_path)
        print('Stored ckpt at {}'.format(ckpt_path))
              




    print('Training complete after {} iters.'.format(itr))
 
    return {
        'avg_section_density_loss': avg_section_density_loss
    }